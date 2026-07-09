import importlib


class BaseAnalyzer:
    """
    Base class for all image and document analyzer plugins.
    """
    VERSION = "1.0.0"
    DEPENDENCIES = []

    def __init__(self, config=None):
        self.config = config or {}

    def get_name(self) -> str:
        """
        Returns the unique name of the analyzer plugin.
        Defaults to the class name in snake_case.
        """
        name = self.__class__.__name__
        # Convert CamelCase to snake_case
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def check_dependencies(self) -> list:
        """
        Checks if all external packages listed in DEPENDENCIES are importable.
        Returns a list of missing package names.
        """
        missing = []
        for dep in self.DEPENDENCIES:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing

    def analyze(self, file_path: str, img, context: dict) -> dict:
        """
        Performs analysis.
        Parameters:
          file_path: Absolute path to the image file.
          img: PIL.Image.Image instance (could be None if image failed to load).
          context: Shared dictionary of previous analyzer outputs (e.g. OCR text).
        Returns:
          A dictionary of results following the schema components.
        """
        raise NotImplementedError("Plugins must implement the analyze method.")
