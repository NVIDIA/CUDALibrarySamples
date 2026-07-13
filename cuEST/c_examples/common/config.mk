# Environment-specific paths for the C examples. Users can override any
# of these on the command line or via the environment.

# Directory containing libxc (the shared or static archive that provides
# xc_func_* symbols). Required for the static-link variant; harmless for
# the shared-link variant whose libxc dep is satisfied via DT_NEEDED.
LIBXC_DIR ?= /usr/local/libxc/lib
