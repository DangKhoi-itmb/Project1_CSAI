import os, sys
print("cwd =", os.getcwd())
print("exists ai_project =", os.path.exists("ai_project"))
print("sys.path =")
for p in sys.path:
    print("   ", p)
