from Scale import SartoriusEntrisScale

scale_port='/dev/ttyACM0'
scale = SartoriusEntrisScale(scale_port)
scale.beep()
scale.reset()
print(scale.get_weight())
input('Change weight and press ENTER to continue')
print(scale.get_weight())