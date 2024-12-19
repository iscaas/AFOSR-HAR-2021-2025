'''
import os
import torch

def get_local_rank():
    # Check if the local rank environment variable is set
    if 'LOCAL_RANK' in os.environ:
        # Get the local rank from the environment variable
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # If the environment variable is not set, assume local rank is 0
        local_rank = 0
    return local_rank

# Get the local rank of the current GPU
local_rank = get_local_rank()

# Print the local rank
print("Local rank:", local_rank)
'''

import site
import os

def find_package_location(package_name):
    # Get the paths to site-packages directories
    site_packages = site.getsitepackages()

    # Search for the package in each site-packages directory
    for site_package in site_packages:
        package_path = os.path.join(site_package, package_name)
        if os.path.exists(package_path):
            return package_path

    # Return None if the package is not found
    return None

# Example usage:
package_name = 'apex'
package_location = find_package_location(package_name)
if package_location:
    print(f"The location of {package_name} is: {package_location}")
else:
    print(f"{package_name} is not installed.")