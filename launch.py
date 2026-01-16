# launch.py – optionaler Launcher für EXE
from streamlit.web import bootstrap
import sys, os

def main():
    file = os.path.join(os.path.dirname(__file__), "app.py")
    bootstrap.run(file, "", [], flag_options={})
if __name__ == "__main__":
    sys.exit(main())
