"""
Image generation module for RetroGraph.

This module handles creating images from trees and molecules.
"""

import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from ..config import IMAGE_WIDTH, IMAGE_HEIGHT
from ..utils.logging import logger

try:
    from aizynthfinder.reactiontree import ReactionTree
except ImportError:
    logger.error("ERROR: AiZynthFinder is required but not installed.")
    logger.error("Please install it and try again.")
    raise ImportError("AiZynthFinder is required but not installed.")

class ImageGenerator:
    """Class for generating images from trees and molecules."""
    
    @staticmethod
    def create_tree_image(
        tree_data: Dict[str, Any],
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Creates a PIL Image of the reaction tree using AiZynthFinder drawing.

        Args:
            tree_data: Tree data dictionary from AiZynthFinder.
            width: Target width for resizing (optional).
            height: Target height for resizing (optional).

        Returns:
            A PIL Image object, or None if image generation fails.
        """
        try:
            reaction_tree = ReactionTree.from_dict(tree_data)
            img = reaction_tree.to_image()

            # Resize only if both width and height are provided
            if width and height and img:
                # Use LANCZOS for high-quality downsampling
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            logger.error(f"Error creating tree image using AiZynthFinder: {e}")
            return None

    @staticmethod
    def create_molecule_image(
        smiles: str,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT
    ) -> Optional[Image.Image]:
        """Generates a PIL Image from a SMILES string using RDKit.

        Args:
            smiles: The SMILES string of the molecule.
            width: Target image width.
            height: Target image height.

        Returns:
            A PIL Image object, or None if generation fails or SMILES is invalid.
        """
        if not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"RDKit could not parse SMILES: {smiles}")
                return None
            # Generate image directly into PIL Image object
            img = Draw.MolToImage(mol, size=(width, height))
            return img
        except Exception as e:
            logger.error(f"Error generating RDKit image for SMILES {smiles}: {e}")
            return None

    @staticmethod
    def image_to_base64_data_url(
        img: Optional[Image.Image],
        format: str = "PNG"
    ) -> str:
        """Converts a PIL Image object to a base64 encoded data URL.

        Args:
            img: The PIL Image object, or None.
            format: The image format (e.g., "PNG", "JPEG").

        Returns:
            A base64 encoded data URL string (e.g., "data:image/png;base64,...").
            Returns an empty string if conversion fails or image is None.
        """
        if not img:
            return ""
        try:
            buffered = io.BytesIO()
            img.save(buffered, format=format)
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/{format.lower()};base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting image to base64 data URL: {e}")
            return "" 