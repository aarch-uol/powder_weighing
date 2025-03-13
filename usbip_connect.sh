#!/bin/bash

sudo modprobe vhci-hcd

sudo usermod -aG dialout $USER

sudo usbip attach -r 10.6.203.8 -b 1-1.1
sudo usbip attach -r 10.6.203.8 -b 1-1.2
