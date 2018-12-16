import matplotlib.pyplot as plt
import numpy as np
import os
import csv

class Graphs():

   def plotGridResults(axis1Name, axis2Name, fileLocation, metric):

       firstAxis = list()
       secondAxis = list()
       metricData = list()
       
       with open(fileLocation) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    for countn in range (len(row)):
                        if (countn - 1) % 3 == 0:
                            words = row[countn].split('_')
                            firstAxis.append(float(words[0]))

                    line_count += 1
                else:
                    for countn in range (len(row)):
                        if (countn == 0):
                           secondAxis.append(float(row[countn]))                          
                        elif (metric == 'accuracy' or metric == 'accuracy2'):
                            if (countn - 3) % 3 == 0 and countn != 0:
                                metricData.append(float(row[countn]))
                        elif (metric == 'timing'):
                            if (countn - 1) % 3 == 0:
                                metricData.append(float(row[countn]))
                    
                    line_count += 1
       print('Finished reading metric data from csv. Building graph..')

       metricData = np.array(metricData)
       metricData=metricData.reshape(len(secondAxis), len(firstAxis))
       fig2, ax2 = plt.subplots(figsize=(12,8))
       c=ax2.contourf(firstAxis,secondAxis,metricData)
       ax2.set_xlabel(axis1Name)
       ax2.set_ylabel(axis2Name)
       fig2.colorbar(c)
       fig2.savefig('%s.png' % metric)
       plt.show()