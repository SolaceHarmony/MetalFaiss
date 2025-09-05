// Minimal bridge header for SwiftFaissC
// This is a placeholder while FAISS C API headers are unavailable

// Platform-specific includes
#ifndef __APPLE__
// On non-Apple platforms, we need basic C headers
#include <stdio.h>
#include <stdlib.h>
#else
// On Apple platforms, we can include Accelerate if needed
#include <Accelerate/Accelerate.h>
#endif

// Placeholder function to make the module compile
void __bridge_dummy(void);
