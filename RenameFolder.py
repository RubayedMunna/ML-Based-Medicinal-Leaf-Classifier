#!/usr/bin/env python
# coding: utf-8

import os

def rename_folders(directory):
    """
    Rename all folders in the specified directory by adding 'Leaf' to their names.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing folders to rename
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return
    
    # Get all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            new_name = item + "Leaf"
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(item_path, new_path)
                print(f"Renamed '{item}' to '{new_name}'")
            except OSError as e:
                print(f"Failed to rename '{item}' to '{new_name}': {e}")
    
    print("Renaming process completed.")

def main():
    # Directory path (update this to your directory)
    directory_path = r"D:\4-2\29(Rubayed)\Machine Learning Lab\Project\MedicinalLeafDataset"
    
    print(f"Starting folder renaming in: {directory_path}")
    rename_folders(directory_path)

if __name__ == "__main__":
    main()
