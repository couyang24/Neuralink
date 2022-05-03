from neuralink._version import __version__
from .logger import get_logger

print("""
Welcome to Neuralink!
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
LOG.info(f'Installing Neuralink Version: {__version__}')
