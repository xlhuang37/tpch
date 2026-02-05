echo off | sudo tee /sys/devices/system/cpu/smt/control
sudo systemctl daemon-reload
sudo systemctl start highperf.slice
sudo systemctl start lowperf.slice
