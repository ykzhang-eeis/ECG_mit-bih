# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks

# - Detect a connected HDK and import the required support package
connected_hdks, support_modules, chip_versions = find_xylo_hdks()

print(connected_hdks)
print(support_modules)
print(chip_versions)