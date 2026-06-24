import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from spellchecker import SpellChecker
from itertools import product

CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EMNIST_CORRECTIONS = {'0': 'o', '8': 'r', '5': 's', '1': 'l', '2': 'z', '6': 'b', '9': 'g'}
confusions = {'b':'d','d':'b','p':'q','q':'p','i':'l','l':'i','1':'l','0':'o'}

try:
    spell = SpellChecker()
except ImportError:
    spell = None

def apply_emnist_correction(chars_str):
    if not chars_str or not any(c.isalpha() for c in chars_str):
        return chars_str
    return "".join([EMNIST_CORRECTIONS.get(c, c) for c in chars_str])

def dyslexia_correction(word):
    if not word:
        return ""
    if not spell:
        return word
    variants = [[c, confusions[c]] if c in confusions else [c] for c in word.lower()]
    variants = [''.join(p) for p in product(*variants)]
    valid = [w for w in variants if spell.correction(w) == w]
    return valid[0] if valid else (spell.correction(word) or word)
