import base64
import os
import sys

if __name__ == "__main__":
    fnames = []
    for filename in os.listdir(sys.argv[1]):
        result = base64.decodestring(filename.encode("utf8"))
        fnames.append((filename, result.decode("utf8")))

    for fname, decoded_fname in sorted(fnames, key=lambda x: x[1]):
        print(decoded_fname, fname.strip())
