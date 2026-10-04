The import no longer runs two warm-up calls that warmed nothing (a scipy call, and a second form of the GLDZM distance map call), so a first import with an empty numba cache compiles one kernel less.
