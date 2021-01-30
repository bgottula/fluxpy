#!/usr/bin/env python3

import sys
import struct
from typing import NamedTuple
import zlib
import sqlite3
import pickle
import numpy as np

NUM_TRACKS = 39
NUM_SECTORS = 12

# each of these starts with 53 1's
# SECTOR_HEADER = np.unpackbits(np.array([0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD, 0x57], dtype=np.uint8))[1:]
# SECTOR_DATA = np.unpackbits(np.array([0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD, 0xDB], dtype=np.uint8))[1:]

# Turns out it's actually better to use less of the leading 1's in the header when matching, since
# then the timing estimate for the start of the sector is better. When the full header is used the
# timing estimate is probably good for the middle of the header, but then it drifts significantly
# towards the edges of the header. When I tried using the full-length header I was getting very
# bad results.
SECTOR_HEADER = np.unpackbits(np.array([0xFF, 0xFD, 0x57], dtype=np.uint8))[1:]
SECTOR_DATA = np.unpackbits(np.array([0xFF, 0xFD, 0xDB], dtype=np.uint8))[1:]
SECTOR_ORDER = [0, 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7]


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
    mean_grc_corr: float


class SectorOrderTracker:
    """Keep track of what sector we're on"""

    def __init__(self, init_sector: int):
        self.set_sector(init_sector)

    def next_sector(self):
        self.sector_order_index = (self.sector_order_index + 1) % NUM_SECTORS
        return SECTOR_ORDER[self.sector_order_index]

    def set_sector(self, sector: int):
        for ii, sec in enumerate(SECTOR_ORDER):
            if sec == sector:
                self.sector_order_index = ii
                break


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


def find_pattern(flux, pattern, threshold=0.8, flag=False):
    """Find the indices where a particular pattern starts"""

    # FIXME: hard-coded
    SYMBOL_PERIOD = 47

    # TODO: determine if we really want to drop the first bit of the diff encoded pattern
    header_bits = diff_encode(pattern)[1:]
    header_symbols = 2 * header_bits.astype(np.float64) - 1
    header_samples = np.repeat(header_symbols, SYMBOL_PERIOD)
    header_samples /= header_samples.size

    conv = np.correlate(flux, header_samples)

    if flag:
        import matplotlib.pyplot as plt
        plt.plot(np.abs(conv))
        plt.grid(True)
        plt.show()
        import IPython; IPython.embed()

    # may get multiple samples above threshold for a single peak
    indices_over_threshold = np.argwhere(np.abs(conv) > threshold)[:,0]

    if indices_over_threshold.size == 0:
        return []

    peak_groups = np.split(indices_over_threshold, np.argwhere(np.diff(indices_over_threshold) > 1)[:,0] + 1)

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


def gcr_decoder(flux, kernels, flag=False):

    # Take absolute value because data is encoded in transitions -- polarity is not important
    corr = np.abs(np.sum(flux * kernels, axis=1))

    if flag:
        print(f'GCR decoder peak: {np.max(corr)}')

    return np.argmax(corr), np.max(corr)


def decode_data_sector(flux, flag=False):

    SYMBOL_PERIOD = 47
    BROTHER_DATA_RECORD_ENCODED_SIZE = 415
    EARLY_LATE_OFFSET = 8

    data_gcr_kernels = make_data_gcr_kernels()
    samples_per_gcr_word = data_gcr_kernels.shape[1]

    decoded_gcr_words = []
    gcr_peaks = []
    flux_start_index = SECTOR_DATA.size * SYMBOL_PERIOD  # discard the segment header
    for _ in range(BROTHER_DATA_RECORD_ENCODED_SIZE):

        flux_early = flux[int(flux_start_index) - EARLY_LATE_OFFSET : int(flux_start_index) + samples_per_gcr_word - EARLY_LATE_OFFSET]
        flux_ontime = flux[int(flux_start_index) : int(flux_start_index) + samples_per_gcr_word]
        flux_late = flux[int(flux_start_index) + EARLY_LATE_OFFSET : int(flux_start_index) + samples_per_gcr_word + EARLY_LATE_OFFSET]

        # early = np.sum(np.abs(np.correlate(flux_early, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))
        # ontime = np.sum(np.abs(np.correlate(flux_ontime, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))
        # late = np.sum(np.abs(np.correlate(flux_late, np.ones(SYMBOL_PERIOD))[::SYMBOL_PERIOD]))

        gcr_word_early, gcr_peak_early = gcr_decoder(flux_early, data_gcr_kernels)
        gcr_word_ontime, gcr_peak_ontime = gcr_decoder(flux_ontime, data_gcr_kernels, flag)
        gcr_word_late, gcr_peak_late = gcr_decoder(flux_late, data_gcr_kernels)

        # adjust timing by one sample forward or backward if needed
        best_seg_idx = np.argmax((gcr_peak_early, gcr_peak_ontime, gcr_peak_late))
        gcr_peak_selected = np.max((gcr_peak_early, gcr_peak_ontime, gcr_peak_late))
        if best_seg_idx == 0:
            gcr_word_selected = gcr_word_early
            flux_start_index -= EARLY_LATE_OFFSET
        elif best_seg_idx == 1:
            gcr_word_selected = gcr_word_ontime
        elif best_seg_idx == 2:
            gcr_word_selected = gcr_word_late
            flux_start_index += EARLY_LATE_OFFSET

        # gcr_word_decoded, gcr_peak = gcr_decoder(flux_selected, data_gcr_kernels, flag)
        decoded_gcr_words.append(gcr_word_selected)
        gcr_peaks.append(gcr_peak_selected)
        flux_start_index += 8 * SYMBOL_PERIOD

    bit_array = np.unpackbits(np.reshape(np.array(decoded_gcr_words, dtype=np.uint8), (-1,1)), axis=1)
    bits = bit_array[:,-5:].flatten()

    gcr_peaks = np.array(gcr_peaks)
    gcr_peak_mean = np.mean(gcr_peaks)
    print(f'mean GCR correlation peak: {gcr_peak_mean}')

    # discard the last byte since this is not actually part of the payload
    return np.packbits(bits)[:-1], gcr_peak_mean

def decode_sector_record(flux):

    # FIXME: hard-coded
    SYMBOL_PERIOD = 47

    # discard the segment header
    flux = flux[SECTOR_HEADER.size * SYMBOL_PERIOD:]

    header_gcr_kernels = make_header_gcr_kernels()
    samples_per_gcr_word = 16 * SYMBOL_PERIOD

    # decode track number
    flux_segment = flux[:samples_per_gcr_word + SYMBOL_PERIOD]
    logical_track, track_gcr_peak = gcr_decoder(flux_segment, header_gcr_kernels[:NUM_TRACKS])

    # decode sector number
    flux_segment = flux[samples_per_gcr_word : 2*samples_per_gcr_word + SYMBOL_PERIOD]
    logical_sector, sector_gcr_peak = gcr_decoder(flux_segment, header_gcr_kernels[:NUM_SECTORS])

    print(f'header decoder GCR peaks: track: {track_gcr_peak} sector: {sector_gcr_peak}')

    # import matplotlib.pyplot as plt
    # import IPython; IPython.embed()

    if logical_track >= NUM_TRACKS:
        # should be impossible to reach this if GCR decoding is implemented correctly
        raise InvalidHeader(f'Logical track is {logical_track} but should be in [0,38]')

    if logical_sector >= NUM_SECTORS:
        # should be impossible to reach this if GCR decoding is implemented correctly
        raise InvalidHeader(f'Logical sector is {logical_sector} but should be in [0,11]')

    # if sector_gcr_peak < 0.6:
        # raise InvalidHeader(f'Sector field of header had low correlation: {sector_gcr_peak}')

    # if track_gcr_peak < 0.6:
        # raise InvalidHeader(f'Track field of header had low correlation: {track_gcr_peak}')

    return logical_track, logical_sector


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

    try:
        cur.execute('SELECT * FROM properties')
    except sqlite3.OperationalError:
        raise RuntimeError(f'This flux file is an old version. Only version 3 is supported.')
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


def filter_header_indices(indices):

    # typical number of samples between consecutive headers is 195000, but can be as high as ~250000
    HEADER_TO_HEADER_SAMPLES_MIN = 190000

    if len(indices) < 2:
        return indices

    if np.all(np.diff(indices) < HEADER_TO_HEADER_SAMPLES_MIN):
        raise ValueError("No indices have expected spacing. Can't handle this situation")

    num_deleted = 0
    while True:
        indices_diff = np.diff(indices)
        if np.all(indices_diff >= HEADER_TO_HEADER_SAMPLES_MIN):
            break

        # work forwards from first good diff
        first_good_diff_index = np.argmax(indices_diff >= HEADER_TO_HEADER_SAMPLES_MIN)
        diffs_after_first_good = indices_diff[first_good_diff_index:]

        if np.all(diffs_after_first_good >= HEADER_TO_HEADER_SAMPLES_MIN):

            # work backward from first good diff
            indices = np.delete(indices, first_good_diff_index - 1)
            num_deleted += 1
            continue

        first_bad_after_good_index = np.argmax(diffs_after_first_good < HEADER_TO_HEADER_SAMPLES_MIN)
        indices = np.delete(indices, first_good_diff_index + first_bad_after_good_index + 1)
        num_deleted += 1

    print(f'Rejecting {num_deleted} header indices that are at unexpected positions')

    return indices


def extract_sectors(tracks):

    # typical number of samples between start of header segment to start of data segment
    HEADER_TO_DATA_SAMPLES_MEAN = 8704

    # allowed range of samples between start of header segment and start of data segment
    HEADER_TO_DATA_SAMPLES_MIN = HEADER_TO_DATA_SAMPLES_MEAN - 300
    HEADER_TO_DATA_SAMPLES_MAX = HEADER_TO_DATA_SAMPLES_MEAN + 300

    # dict of DataSegment objects where key is a tuple of (logical_track, logical_segment)
    sectors = {}

    for ii, track in enumerate(tracks):
        print(f'Decoding flux for physical track {track.track}...')

        header_start_indices = find_pattern(track.flux, SECTOR_HEADER)
        data_start_indices = find_pattern(track.flux, SECTOR_DATA)

        header_start_indices = filter_header_indices(header_start_indices)

        print(f'Found {len(header_start_indices)} candidate headers and {len(data_start_indices)} candidate data segments')

        sector_order_tracker = None
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

            # FIXME: should use track.track, not rely on loop index to check physical track number
            # sanity check logical track number
            # if len(tracks) == 40:
            #     if ii != logical_track:
            #         print(f'Warning: expected logical track {ii} but got {logical_track}; skipping this header')
            #         import IPython; IPython.embed()
            #         continue
            # elif len(tracks) == 82:
            #     if (ii - 1) // 2 != logical_track:
            #         print(f'Warning: expected logical track {(ii - 1) // 2} but got {logical_track}; skipping this header')
            #         import IPython; IPython.embed()
            #         continue

            # sanity check sector order
            if sector_order_tracker is None:
                sector_order_tracker = SectorOrderTracker(logical_sector)
            else:
                expected_sector = sector_order_tracker.next_sector()
                if expected_sector != logical_sector:
                    print(f'Warning: Expected logical sector {expected_sector} but got {logical_sector}; repairing')
                    logical_sector = expected_sector
                    # print(f'Warning: Expected logical sector {expected_sector} but got {logical_sector}; skipping this header')
                    # import IPython; IPython.embed()
                    # continue

            sector_dict_key = (logical_track, logical_sector)

            # import matplotlib.pyplot as plt
            # import IPython; IPython.embed()

            # TODO: Remove me!
            # sectors[sector_dict_key] = DataSector(
            #     logical_track,
            #     logical_sector,
            #     b'\x00' * 256,
            #     False,
            #     0.0
            # )
            # continue


            data_start_candidates = data_start_indices[np.logical_and(
                data_start_indices >= header_start_index + HEADER_TO_DATA_SAMPLES_MIN,
                data_start_indices <= header_start_index + HEADER_TO_DATA_SAMPLES_MAX)]

            if data_start_candidates.size == 0:
                print(f'No data segments found for header starting at {header_start_index} on physical track {track.track}; possibly off end of flux file')
                # if header_start_index < (track.flux.size - HEADER_TO_DATA_SAMPLES_MAX):
                #     import IPython; IPython.embed()
                continue

            for data_start_index in data_start_candidates:

                try:
                    data_bytes, mean_gcr_peak = decode_data_sector(track.flux[data_start_index - 47:])
                    print(f'decoded data segment for logical track: {logical_track} logical sector: {logical_sector} crc pass: {check_crc(data_bytes)}')
                except ValueError:
                    print(f'Could not decode data starting at {data_start_index} on physical track {track.track}; probably off end of flux file')
                    continue

                if sector_dict_key not in sectors:
                    sectors[sector_dict_key] = DataSector(
                        logical_track,
                        logical_sector,
                        data_bytes[:256].tobytes(),
                        check_crc(data_bytes),
                        mean_gcr_peak
                    )
                else:
                    sector = sectors[sector_dict_key]

                    # only replace existing sector if the new one passes CRC check but the exiting
                    # one doesn't, or if neither passes CRC but the new one had higher mean GCR
                    # correlation peaks whigh may indicate that fewer errors are present
                    if not sector.crc_pass:
                        if check_crc(data_bytes) or mean_gcr_peak > sector.mean_gcr_peak:
                            print(f'replacing data segment for logical track: {logical_track} logical sector: {logical_sector}')
                            sectors[sector_dict_key] = DataSector(
                                logical_track,
                                logical_sector,
                                data_bytes[:256].tobytes(),
                                check_crc(data_bytes),
                                mean_gcr_peak
                            )

    return sectors


def main(filename):

    try:
        tracks = pickle.load(open(filename + '.pickle', 'rb'))
        print('Loaded decoded flux from pickle file')
    except:
        print('Loading flux from fluxengine flux file...')
        tracks = load_flux_file(filename)
        pickle.dump(tracks, open(filename + '.pickle', 'wb'))

    try:
        sectors = pickle.load(open(filename + '.sectors.pickle', 'rb'))
        print('Loaded sectors from pickle file')
    except:
        print('Extracting sectors from flux...')
        sectors = extract_sectors(tracks)
        pickle.dump(sectors, open(filename + '.sectors.pickle', 'wb'))


    # assemble contents of the disk into a monolithic byte array
    disk_image = b''

    missing_sectors = 0
    bad_crcs = 0
    for track in range(39):
        for sector in range(12):
            try:
                disk_image += sectors[(track, sector)].payload
            except KeyError:
                print(f'Adding track {track:02d} sector {sector:02d}: not found; filling with zeros')
                disk_image += b'\x00' * 256
                missing_sectors += 1
            else:
                print(f'Adding track {track:02d} sector {sector:02d} crc {sectors[(track, sector)].crc_pass}')
                if not sectors[(track, sector)].crc_pass:
                    bad_crcs += 1

    print(f'Summary: {bad_crcs} bad CRCs, {missing_sectors} missing sectors out of {39*12} total sectors')

    with open(filename + '.img', 'wb') as f:
        f.write(disk_image)

if __name__ == "__main__":
    main(sys.argv[1])
