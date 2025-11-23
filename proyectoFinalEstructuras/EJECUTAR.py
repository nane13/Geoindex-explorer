import os
import subprocess
import sys

def main():
    # Ruta absoluta del archivo principal de la app
    ruta_app = os.path.join(os.path.dirname(__file__), "run_app.py")

    # Ejecutar streamlit
    comando = [sys.executable, "-m", "streamlit", "run", ruta_app]

    print("Iniciando aplicación Streamlit...\n")
    subprocess.run(comando)

if __name__ == "__main__":
    main()
