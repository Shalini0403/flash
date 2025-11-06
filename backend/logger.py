"""
Logger Module
Provides consistent logging utilities for the Flashcard Generator API
"""


class Logger:
    """Simple logging utility for consistent output"""
    
    @staticmethod
    def header(message: str, width: int = 80):
        """
        Print a header message with border.
        
        Args:
            message: The message to display
            width: Width of the border (default: 80)
        """
        print("\n" + "=" * width)
        print(message)
        print("=" * width)
    
    @staticmethod
    def section(message: str):
        """
        Print a section message.
        
        Args:
            message: The section message to display
        """
        print(f"\n{message}")
    
    @staticmethod
    def info(message: str, indent: int = 2):
        """
        Print an info message.
        
        Args:
            message: The info message to display
            indent: Number of spaces to indent (default: 2)
        """
        print(" " * indent + f"[INFO] {message}")
    
    @staticmethod
    def success(message: str, indent: int = 2):
        """
        Print a success message.
        
        Args:
            message: The success message to display
            indent: Number of spaces to indent (default: 2)
        """
        print(" " * indent + f"[OK] {message}")
    
    @staticmethod
    def warning(message: str, indent: int = 2):
        """
        Print a warning message.
        
        Args:
            message: The warning message to display
            indent: Number of spaces to indent (default: 2)
        """
        print(" " * indent + f"[WARNING] {message}")
    
    @staticmethod
    def error(message: str, indent: int = 2):
        """
        Print an error message.
        
        Args:
            message: The error message to display
            indent: Number of spaces to indent (default: 2)
        """
        print(" " * indent + f"[ERROR] {message}")
    
    @staticmethod
    def debug(message: str, indent: int = 2):
        """
        Print a debug message.
        
        Args:
            message: The debug message to display
            indent: Number of spaces to indent (default: 2)
        """
        print(" " * indent + f"[DEBUG] {message}")
    
    @staticmethod
    def divider(width: int = 80, char: str = "-"):
        """
        Print a divider line.
        
        Args:
            width: Width of the divider (default: 80)
            char: Character to use for the divider (default: -)
        """
        print(char * width)