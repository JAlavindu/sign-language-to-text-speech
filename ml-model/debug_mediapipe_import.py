import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import mediapipe
    print(f"Mediapipe file: {mediapipe.__file__}")
    print(f"Mediapipe path: {mediapipe.__path__}")
    print(f"Has solutions? {'solutions' in dir(mediapipe)}")
    
    try:
        import mediapipe.python.solutions
        print("Imported mediapipe.python.solutions successfully")
    except ImportError as e:
        print(f"Failed to import mediapipe.python.solutions: {e}")
        
except ImportError as e:
    print(f"Failed to import mediapipe: {e}")
