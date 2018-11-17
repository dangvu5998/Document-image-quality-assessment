import argparse
from .utils import generate_patches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='Source image file, contains good and bad')
    parser.add_argument('--dist', help='Destination directory for data')