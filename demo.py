import depthai

print("SSS")

devices = depthai.Device.getAllAvailableDevices()
print(devices)
for d in devices:
    print(d.desc.name)