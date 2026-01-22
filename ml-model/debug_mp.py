import sys
print(sys.executable)
try:
    import mediapipe.python.solutions
    print("Direct import of solutions successful")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    import mediapipe as mp
    print(f"Mediapipe file: {mp.__file__}")
    print(dir(mp))
    print("Solutions:", mp.solutions)
except Exception as e:
    print(e)
