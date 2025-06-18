import ctypes

# Load the shared library
lib = ctypes.CDLL("./libops.so")

# Define the function signature: void (*)(int)
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int)

# Lookup C functions
f1 = CALLBACK(("f1", lib))
f2 = CALLBACK(("f2", lib))
print(f2(10))

# Lookup dispatcher
call_with_callback = lib.call_with_callback
call_with_callback.argtypes = [CALLBACK]

# Select one at runtime
chosen = f2  # dynamically choose which function

# Call
call_with_callback(f2)
