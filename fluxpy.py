#!/usr/bin/env python3

import sys
import struct
from typing import NamedTuple
import zlib
import sqlite3
import pickle
import numpy as np


SECTOR_HEADER = np.unpackbits(np.array([0xFF, 0xFF, 0xFD, 0x57], dtype=np.uint8))
SECTOR_DATA = np.unpackbits(np.array([0xFF, 0xFF, 0xFD, 0xDB], dtype=np.uint8))

class InvalidHeader(Exception):
    """Raised when an invalid header field is encountered"""

class FluxTrack(NamedTuple):
    track: int
    flux: np.ndarray

class DataSector(NamedTuple):
    logical_track: int
    logical_sector: int
    payload: bytes
    crc_pass: bool


def bytecode_to_array(bytecode: bytes):
    """Converts FluxEngine bytecode format to a numpy array giving pulse events versus time.

    The bytecode format is described in one sentence in the project documentation:

    "The bytecode format is very simple with a six-bit interval since the previous event in the
    lower six bits and the top two bits are set of a pulse or an index hole (or both, or neither)."

    Source: http://cowlark.com/fluxengine/doc/technical.html

    From the project source code we find more specifically that:
    - Pulse event is indicated when the MSB is set (bit 7)
    - Index hole event is indicated when the second bit is set (bit 6)

    Source: https://github.com/davidgiven/fluxengine/blob/master/lib/fluxmap.cc

    Args:
        bytecode: bytes array from a .flux file

    Returns:
        A tuple containing:
        1) A Numpy array where the sample rate is 12 MHz (set by the FluxEngine hardware) and each
           element in the array is either 1 or 0 where 1 indicates that a pulse event occurred
           during that clock period.
        2) A Numpy array with the bipodal level of the flux versus time. The raw pulse information
           only encodes the transitions so the polarity of this output is arbitrary. The sample
           rate is 12 MHz.
        3) A Numpy array giving the interval in seconds between each pulse.
    """
    TICK_PERIOD = 1 / 12e6  # 12 MHz clock
    pulse_intervals = []
    pulse_train = []
    flux = []
    flux_state = False
    ticks_since_last_pulse = 0

    for b in bytecode:
        pulse = b & 0x80
        ticks_since_last_pulse += b & 0x3F
        if pulse:
            pulse_intervals.append(ticks_since_last_pulse * TICK_PERIOD)
            pulse_train += [0] * ticks_since_last_pulse
            pulse_train.append(1)
            flux += [2 * float(flux_state) - 1] * ticks_since_last_pulse
            flux_state = not flux_state
            ticks_since_last_pulse = 0

    return np.array(pulse_train, dtype=np.uint8), np.array(flux), np.array(pulse_intervals)


def diff_encode(input):

    out = [0]
    for bit in input:
        if bit:
            out.append(1 - out[-1])
        else:
            out.append(out[-1])

    return np.array(out, dtype=np.uint8)


def find_pattern(flux, pattern, threshold=0.85):
    """Find the indices where a particular pattern starts"""

    # FIXME: hard-coded
    SYMBOL_PERIOD = 47

    # TODO: determine if we really want to drop the first bit of the diff encoded pattern
    header_bits = diff_encode(pattern)[1:]
    header_symbols = 2 * header_bits.astype(np.float64) - 1
    header_samples = np.repeat(header_symbols, SYMBOL_PERIOD)
    header_samples /= header_samples.size

    conv = np.correlate(flux, header_samples)

    # may get multiple samples above threshold for a single peak
    indices_over_threshold = np.argwhere(np.abs(conv) > threshold)[:,0]

    if indices_over_threshold.size == 0:
        return []

    peak_groups = np.split(indices_over_threshold, np.argwhere(np.diff(indices_over_threshold) > 4)[:,0] + 1)

    peak_indices = []
    for peak_group_indices in peak_groups:
        group_peak_index = np.argmax(np.abs(conv[peak_group_indices]))
        peak_indices.append(peak_group_indices[0] + group_peak_index)

    return np.array(peak_indices)


def make_matched_filter_from_bit_pattern(bit_pattern):

    # FIXME: hard-coded
    SYMBOL_PERIOD = 47

    bits_encoded = diff_encode(bit_pattern)
    symbols = 2 * bits_encoded.astype(np.float64) - 1
    samples = np.repeat(symbols, SYMBOL_PERIOD)
    samples /= samples.size

    return samples


def make_header_gcr_kernels():

    encoded_bytes = [
        [0xDF, 0xB5],  # 0
        [0x5B, 0x6F],  # 1
        [0x7D, 0xF7],  # 2
        [0xBF, 0xD5],  # 3
        [0xF5, 0x7F],  # 4
        [0x6D, 0x5D],  # 5
        [0xAF, 0xEB],  # 6
        [0xDD, 0xB7],  # 7
        [0x57, 0x75],  # 8
        [0x7B, 0xFB],  # 9
        [0xBD, 0xD7],  # 10
        [0xEF, 0xAB],  # 11
        [0x6B, 0x5F],  # 12
        [0xAD, 0xED],  # 13
        [0xDB, 0xBB],  # 14
        [0x55, 0x77],  # 15
        [0x77, 0xDB],  # 16
        [0xBB, 0xAD],  # 17
        [0xED, 0x6B],  # 18
        [0x5F, 0xEF],  # 19
        [0xAB, 0xBD],  # 20
        [0xD7, 0x7B],  # 21
        [0xFB, 0x57],  # 22
        [0x75, 0xDD],  # 23
        [0xB7, 0xAF],  # 24
        [0xEB, 0x6D],  # 25
        [0x5D, 0xF5],  # 26
        [0x7F, 0xBF],  # 27
        [0xD5, 0x7D],  # 28
        [0xF7, 0x5B],  # 29
        [0x6F, 0xDF],  # 30
        [0xB5, 0xB5],  # 31
        [0xDF, 0x6F],  # 32
        [0x5B, 0xF7],  # 33
        [0x7D, 0xD5],  # 34
        [0xBF, 0x7F],  # 35
        [0xF5, 0x5D],  # 36
        [0x6D, 0xEB],  # 37
        [0xAF, 0xB7],  # 38
        [0xDD, 0x75],  # 39
        [0x57, 0xFB],  # 40
        [0x7B, 0xD7],  # 41
        [0xBD, 0xAB],  # 42
        [0xEF, 0x5F],  # 43
        [0x6B, 0xED],  # 44
        [0xAD, 0xBB],  # 45
        [0xDB, 0x77],  # 46
        [0xBB, 0x55],  # 47
        [0xED, 0xDB],  # 48
        [0x5F, 0xAD],  # 49
        [0xAB, 0x6B],  # 50
        [0xD7, 0xEF],  # 51
        [0xFB, 0xBD],  # 52
        [0x75, 0x7B],  # 53
        [0xB7, 0x57],  # 54
        [0xEB, 0xDD],  # 55
        [0x5D, 0xAF],  # 56
        [0x7F, 0x6D],  # 57
        [0xD5, 0xF5],  # 58
        [0xF7, 0xBF],  # 59
        [0x6F, 0x7D],  # 60
        [0xB5, 0x5B],  # 61
        [0xDF, 0xDF],  # 62
        [0x5B, 0xB5],  # 63
        [0x7D, 0x6F],  # 64
        [0xBF, 0xF7],  # 65
        [0xF5, 0xD5],  # 66
        [0x6D, 0x7F],  # 67
        [0xAF, 0x5D],  # 68
        [0xDD, 0xEB],  # 69
        [0x57, 0xB7],  # 70
        [0x7B, 0x75],  # 71
        [0xBD, 0xFB],  # 72
        [0xEF, 0xD7],  # 73
        [0x6B, 0xAB],  # 74
        [0xAD, 0x5F],  # 75
        [0xDB, 0xED],  # 76
        [0x55, 0xBB],  # 77
    ]

    kernels = []
    for byte_list in encoded_bytes:
        kernels.append(
            make_matched_filter_from_bit_pattern(np.unpackbits(np.array(byte_list, dtype=np.uint8)))
        )

    return np.stack(kernels)


def make_data_gcr_kernels():

    encoded_bytes = [
        0x55,  # 00000
        0x57,  # 00001
        0x5b,  # 00010
        0x5d,  # 00011
        0x5f,  # 00100
        0x6b,  # 00101
        0x6d,  # 00110
        0x6f,  # 00111
        0x75,  # 01000
        0x77,  # 01001
        0x7b,  # 01010
        0x7d,  # 01011
        0x7f,  # 01100
        0xab,  # 01101
        0xad,  # 01110
        0xaf,  # 01111
        0xb5,  # 10000
        0xb7,  # 10001
        0xbb,  # 10010
        0xbd,  # 10011
        0xbf,  # 10100
        0xd5,  # 10101
        0xd7,  # 10110
        0xdb,  # 10111
        0xdd,  # 11000
        0xdf,  # 11001
        0xeb,  # 11010
        0xed,  # 11011
        0xef,  # 11100
        0xf5,  # 11101
        0xf7,  # 11110
        0xfb,  # 11111
    ]

    kernels = []
    for byte in encoded_bytes:
        kernels.append(
            make_matched_filter_from_bit_pattern(np.unpackbits(np.array([byte], dtype=np.uint8)))
        )

    return np.stack(kernels)


def gcr_decoder(flux, kernels):

    # Take absolute value because data is encoded in transitions -- polarity is not important
    corr = np.abs(np.sum(flux * kernels, axis=1))
    return np.argmax(corr)


def decode_data_sector(flux):

    SYMBOL_PERIOD = 47

    BROTHER_DATA_RECORD_ENCODED_SIZE = 415

    data_gcr_kernels = make_data_gcr_kernels()
    samples_per_gcr_word = data_gcr_kernels.shape[1]

    decoded_gcr_words = []
    flux_start_index = 32*SYMBOL_PERIOD  # discard the segment header
    for _ in range(BROTHER_DATA_RECORD_ENCODED_SIZE):

        flux_early = flux[int(flux_start_index) - 1 : int(flux_start_index) + samples_per_gcr_word - 1]
        flux_ontime = flux[int(flux_start_index) : int(flux_start_index) + samples_per_gcr_word]
        flux_late = flux[int(flux_start_index) + 1 : int(flux_start_index) + samples_per_gcr_word + 1]

        early = np.sum(np.abs(np.correlate(flux_early, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))
        ontime = np.sum(np.abs(np.correlate(flux_ontime, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))
        late = np.sum(np.abs(np.correlate(flux_late, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))

        # adjust timing by one sample forward or backward if needed
        best_seg_idx = np.argmax((early, ontime, late))
        if best_seg_idx == 0:
            flux_selected = flux_early
            flux_start_index -= 1.0
        elif best_seg_idx == 1:
            flux_selected = flux_ontime
        elif best_seg_idx == 2:
            flux_selected = flux_late
            flux_start_index += 1.0

        decoded_gcr_words.append(gcr_decoder(flux_selected, data_gcr_kernels))
        flux_start_index += 8*SYMBOL_PERIOD

    bit_array = np.unpackbits(np.reshape(np.array(decoded_gcr_words, dtype=np.uint8), (-1,1)), axis=1)
    bits = bit_array[:,-5:].flatten()

    # discard the last byte since this is not actually part of the payload
    return np.packbits(bits)[:-1]

def decode_sector_record(flux):

    # FIXME: hard-coded
    SYMBOL_PERIOD = 47

    # discard the segment header
    flux = flux[32*SYMBOL_PERIOD:]

    header_gcr_kernels = make_header_gcr_kernels()
    samples_per_gcr_word = 16 * SYMBOL_PERIOD

    decoded_gcr_words = []
    for ii in range(2):
        flux_segment = flux[ii*samples_per_gcr_word : (ii+1)*samples_per_gcr_word + SYMBOL_PERIOD]
        decoded_gcr_words.append(gcr_decoder(flux_segment, header_gcr_kernels))

    logical_track = decoded_gcr_words[0]
    logical_sector = decoded_gcr_words[1]

    if logical_track > 38:
        raise InvalidHeader(f'Logical track is {logical_track} but should be in [0,38]')

    if logical_sector > 11:
        raise InvalidHeader(f'Logical sector is {logical_sector} but should be in [0,11]')

    return decoded_gcr_words


def compute_crc(data_sector):

    BROTHER_POLY = 0x000201

    crc = data_sector[0]
    for byte in data_sector[1:256]:
        for _ in range(8):
            crc = ((crc << 1) ^ BROTHER_POLY) if (crc & 0x800000) else (crc << 1)
        crc ^= byte

    return crc & 0xFFFFFF


def check_crc(data_sector):

    crc_in_record = struct.unpack('>I', b'\x00' + data_sector[-3:].tobytes())[0]
    crc_from_data = compute_crc(data_sector)

    return crc_in_record == crc_from_data


def load_flux_file(filename):

    conn = sqlite3.connect(filename)
    cur = conn.cursor()

    cur.execute('SELECT * FROM properties')
    properties = cur.fetchall()

    for p in properties:
        if p[0] == 'version':
            if p[1] != '3':
                raise RuntimeError(f'This flux file is version {p[1]} but only version 3 is supported')

    cur.execute('SELECT * FROM zdata')
    data = cur.fetchall()

    tracks = []
    for ii, track in enumerate(data):
        print(f'loading track {ii:02d} of {len(data)}')
        tracks.append(
            FluxTrack(
                track[0],  # track number
                bytecode_to_array(zlib.decompress(track[2]))[1],  # decoded flux data
            )
        )

    return tracks



"""
Things left to do:

- read the full .flux file from SQLITE database and extract all the sectors
+ symbol timing recovery
+ identify start of each sector header segment and sector data segment
+ differential decoding (1's are encoded as flux transitions)
+ GCR decoding with soft decision input and error correction for both header and data
  > fluxengine decodes GRC with a lookup table which means it just fails if there are any bit
    errors in the raw flux transitions.
"""


def main(filename):

    # typical number of samples between start of header segment to start of data segment
    HEADER_TO_DATA_SAMPLES_MEAN = 8704

    # allowed range of samples between start of header segment and start of data segment
    HEADER_TO_DATA_SAMPLES_MIN = HEADER_TO_DATA_SAMPLES_MEAN - 100
    HEADER_TO_DATA_SAMPLES_MAX = HEADER_TO_DATA_SAMPLES_MEAN + 100

    try:
        tracks = pickle.load(open(filename + '.pickle', 'rb'))
        print('Loaded from pickle file!')
    except:
        print('No pickle found, loading from flux file...')
        tracks = load_flux_file(filename)
        print('Pickling this to speed up future load')
        pickle.dump(tracks, open(filename + '.pickle', 'wb'))

    # dict of DataSegment objects where key is a tuple of (logical_track, logical_segment)
    sectors = {}

    for track in tracks:
        print(f'Decoding flux for physical track {track.track}...')

        header_start_indices = find_pattern(track.flux, SECTOR_HEADER)
        data_start_indices = find_pattern(track.flux, SECTOR_DATA)

        for header_start_index in header_start_indices:
            try:
                logical_track, logical_sector = decode_sector_record(track.flux[header_start_index - 47:])
            except ValueError:
                print(f'Could not decode header starting at {header_start_index} on physical track {track.track}; probably off end of flux file')
                continue
            except InvalidHeader as e:
                print('Discarding invalid header: ' + str(e))
                continue

            print(f'found header for logical track: {logical_track} logical sector: {logical_sector}')
            sector_dict_key = (logical_track, logical_sector)

            data_start_candidates = data_start_indices[np.logical_and(
                data_start_indices >= header_start_index + HEADER_TO_DATA_SAMPLES_MIN,
                data_start_indices <= header_start_index + HEADER_TO_DATA_SAMPLES_MAX)]

            if data_start_candidates.size == 0:
                print(f'No data segments found for header starting at {header_start_index} on physical track {track.track}; probably off end of flux file')
                continue

            for data_start_index in data_start_candidates:

                try:
                    data_bytes = decode_data_sector(track.flux[data_start_index - 47:])
                except ValueError:
                    print(f'Could not decode data starting at {data_start_index} on physical track {track.track}; probably off end of flux file')
                    continue

                if sector_dict_key not in sectors:
                    print(f'decoded new data segment for logical track: {logical_track} logical sector: {logical_sector}')
                    sectors[sector_dict_key] = DataSector(
                        logical_track,
                        logical_sector,
                        data_bytes[:256],
                        check_crc(data_bytes),
                    )
                else:
                    sector = sectors[sector_dict_key]

                    # only replace existing sector if the new one passes CRC check but the exiting
                    # one doesn't
                    if not sector.crc_pass and check_crc(data_bytes):
                        print(f'replacing data segment for logical track: {logical_track} logical sector: {logical_sector}')
                        sectors[sector_dict_key] = DataSector(
                            logical_track,
                            logical_sector,
                            data_bytes[:256],
                            check_crc(data_bytes),
                        )



    # with open(filename, 'rb') as f:
    #     data_zlib = f.read()
    #     data = zlib.decompress(data_zlib)

    # pulse_train, flux, pulse_intervals = bytecode_to_array(data)

    # header_start_indices = find_pattern(flux, SECTOR_HEADER)
    # data_start_indices = find_pattern(flux, SECTOR_DATA)

    # header_bytes_pos = []
    # header_bytes_neg = []
    # sector_0_idx = None
    # for header_start_index in header_start_indices:
    #     flux_sector = flux[header_start_index - 47:]
    #     header_bytes_pos.append(decode_sector_record(flux_sector))
    #     header_bytes_neg.append(decode_sector_record(1 - flux_sector))
    #     if header_bytes_pos[-1][1] == 0 and sector_0_idx is None:
    #         sector_0_idx = len(header_bytes_pos) - 1

    # flux_data_sector_0 = flux[data_start_indices[sector_0_idx] - 47 :]
    # data_bytes_sector_0 = decode_data_sector(flux_data_sector_0)
    # payload_bytes_sector_0 = data_bytes_sector_0[:256]

    # if not check_crc(data_bytes_sector_0):
    #     print('CRC check failed')

    # payload_bytes_sector_0.tofile('/tmp/good_disk/track0sector0.bin')

    import IPython; IPython.embed()

if __name__ == "__main__":
    main(sys.argv[1])
