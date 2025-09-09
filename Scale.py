from time import time, sleep
import serial
import re
from abc import ABC, abstractmethod

class Scale(ABC):
	
	def __init__(self, scale_port):
		self.serial_com = serial.Serial(
			port=scale_port,
			baudrate=9600,
		)
		self._weight=0
	
	@abstractmethod
	def get_weight(self):
		pass

	@abstractmethod
	def reset(self):
		pass
	
class SartoriusEntrisScale(Scale):
	
	def get_weight(self):
		# empty buffer
		while self.serial_com.in_waiting > 0:
			data = self.serial_com.readline().decode('utf-8').strip()
		# print
		self.serial_com.write(bytearray('!P\r\n', 'ascii'))
		# wait for input
		# check every half second if the measurement has been done
		while self.serial_com.in_waiting<=0:
			sleep(0.5)
		data = self.serial_com.readline().decode('utf-8').strip()
		data = re.search(r"[-+]?\d*\.?\d+", data).group()
		data = float(data)
		return data
		
	def beep(self):
		self.serial_com.write(b'!Q\r\n')
	
	def reset(self):
		self._weight=0
		# send tar command
		self.serial_com.write(bytearray('!T\r\n', 'ascii'))

class FisherScale(Scale):
	
	def __read_weight_for_seconds(self,measurement_time=5):
		#  empty buffer 
		while self.serial_com.in_waiting > 0:
			data = self.serial_com.readline().decode('utf-8').strip()
		# set end time 10s from now
		t_end = time.time() + measurement_time
		weight_points = []
		while time.time() < t_end:
			data = self.serial_com.readline().decode('utf-8').strip()
			data=data.replace('?', '')
			data = int(data)
			weight_points.append(data)
		return weight_points

	def get_weight(self, measurement_time=5):
		""" 
			Reads the incoming weight through serial
			Reads incoming traffic for measured_time seconds, and returns 
			the most frequent value greater or equalt to the last seen value.
			If there is no such value than the greatest seen value is 
			returned
		"""	
		
		weight_points=self.__read_weight_for_seconds(measurement_time)
		# get all the weights seen that were larger than the previous one
		filtered_weights = [ weight for weight in weight_points if weight>self._weight]
		# if possible take the most frequent value greater than prev value
		if len(filtered_weights)!=0:
			measured_weight = max(set(weight_points), key=weight_points.count)
			self._weight = measured_weight
			return measured_weight
		# otherwise all weights were smaller 
		max_weight = max(weight_points)
		read_counter = 0 
		while len(filtered_weights)==0 and abs(self._weight-max_weight)>2:
			weight_points=self.__read_weight_for_seconds(measurement_time)
			max_weight = max(weight_points)
			read_counter+=1
			# if it's done more than 10 readings stop it
			if read_counter>10:
				max_weight = self._weight-2
				break
		measured_weight = max_weight
		self._weight = measured_weight
		return measured_weight	
			
		
	def reset(self):
		self._weight=0

	