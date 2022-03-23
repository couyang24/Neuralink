from DNAUtilities._version import __version__
from .logger import get_logger

print("""
Welcome to deeplearning!
            .
            ! r,|`
             ]r'|
           _gggg$g_
      `,zzm$MMMMMM$mz'~'
     ' ~~~~$MWCPUM$@~~~
       .`-+$MMMMMM$+-`..
            *$FQTP
             1r |.
            |'L" |
             '
""")

LOG = get_logger(__name__)
LOG.info(f'Installing deeplearning Version: {__version__}')
