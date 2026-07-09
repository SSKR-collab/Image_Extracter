from PIL import Image

class PerceptualHash:
    """
    Implements Average Hash (aHash) and Difference Hash (dHash) 
    using pure Python and Pillow.
    """

    @staticmethod
    def _get_resample_filter():
        # Handle Pillow compatibility for Resampling filter names
        try:
            return Image.Resampling.LANCZOS
        except AttributeError:
            try:
                return Image.LANCZOS
            except AttributeError:
                return Image.BILINEAR

    @classmethod
    def ahash(cls, img: Image.Image) -> str:
        """
        Average Hash (aHash)
        """
        try:
            # 1. Grayscale and resize to 8x8
            resample = cls._get_resample_filter()
            gray = img.convert("L").resize((8, 8), resample)
            pixels = list(gray.getdata())
            
            # 2. Calculate average
            avg = sum(pixels) / 64.0
            
            # 3. Build hash representation (64 bits)
            bits = 0
            for i, p in enumerate(pixels):
                if p >= avg:
                    bits |= (1 << (63 - i))
            
            # 4. Form hex string
            return f"{bits:016x}"
        except Exception:
            return None

    @classmethod
    def dhash(cls, img: Image.Image) -> str:
        """
        Difference Hash (dHash)
        """
        try:
            # 1. Grayscale and resize to 9x8 (9 width, 8 height)
            resample = cls._get_resample_filter()
            gray = img.convert("L").resize((9, 8), resample)
            pixels = list(gray.getdata())
            
            # 2. Compare adjacent pixels
            bits = 0
            bit_index = 0
            for y in range(8):
                for x in range(8):
                    left = pixels[y * 9 + x]
                    right = pixels[y * 9 + (x + 1)]
                    if left > right:
                        bits |= (1 << (63 - bit_index))
                    bit_index += 1
            
            # 3. Form hex string
            return f"{bits:016x}"
        except Exception:
            return None
