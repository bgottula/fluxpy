This was written to decode some Brother WP-1 floppy disks which use a
proprietary format. The disks were read using
[FluxEngine](http://cowlark.com/fluxengine/index.html), an open-source and
open-hardware project. FluxEngine struggled with certain disks that were
written to the wrong media type. This script reads in a .flux file generated
by FluxEngine, which contains a representation of the raw flux transitions
read from the disk drive, and uses a different set of algorithms to recover
the data encoded therein. This was able to fully recover several disks that
were only partially recoverable using FluxEngine's decoder.
