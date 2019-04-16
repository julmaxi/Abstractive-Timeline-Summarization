import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

data="""
S , exploded in the Gulf on April 20 . 2010-5-27
Bromwich is head of the office that succeeded the Minerals Management Agency blamed with not d
oing enough inspections on oil platforms like the Deepwater Horizon rig that exploded on April
 20 and created the worst environmental disaster in U.S. history . 2010-10-12
He has played at least seven times since the Deepwater Horizon rig exploded on April 20 , crea
ting America 's biggest environmental disaster . 2010-6-28
BP is worth $ 75 billion less on the open market than it was when the Deepwater Horizon rig ex
ploded six weeks ago . 2010-6-3
Eleven rig workers died when the Deepwater Horizon rig exploded and sank six weeks ago . 2010-6-1
Nearly 5m barrels of oil have gushed into the ocean since the Deepwater Horizon rig exploded a
nd sank in April , according to federal scientists . 2010-8-3
Energy firm beats expectations with # 3.6 bn quarterly profit but performance is overshadowed 
by Deepwater Horizon rig disaster Tuesday 27 April 2010 08.47 BST The Deepwater Horizon rig bu
rning last week . 2010-4-27
The issue troubling Buchanan is that Cuba is looking to drill for oil even deeper than BP 's D
eepwater Horizon rig , which exploded last April and sent millions of barrels of oil into the 
Gulf . 2011-1-24
It all started when an underwater rig exploded in April . 2010-9-8
The Deepwater Horizon drilling rig exploded in the Gulf of Mexico on April 20 after a well rup
tured . 2010-5-6
The Deepwater Horizon drilling rig exploded in April after an uncontrolled surge of oil and ga
s went up the well . 2010-10-12
The two senators had previously lobbied Transocean , owner of the Deepwater Horizon rig that e
xploded on 20 April , against paying dividends to its own shareholders . 2010-6-3
When the Deepwater Horizon oil rig exploded on April 20 , Joye was researching methane leakage
 from oil drilling aboard a National Oceanic and Atmospheric Agency-funded research vessel 8 m
iles from the disaster site . 2010-6-8
As we have all discovered , after the Deepwater Horizon rig exploded on 20 April , the company
 had no systems in place to effectively respond to this scenario . 2010-6-19
Lawmakers are understandably disgusted by the environmental damage wreaked since BP 's Deepwat
er Horizon oil rig exploded and sank on April 20 . 2010-6-17
2.45 pm : Investors are asking for more details about exactly what went wrong on the Deepwater
 Horizon before it exploded on 20 April , but Hayward is n't biting . 2010-6-4
Shares in BP rose by 1.5 % this morning to 430p , around a third lower than their value before
 the Deepwater Horizon rig exploded in April . 2010-11-2
The Deepwater Horizon rig - operated by Transocean on behalf of BP and its partners - exploded
 on 20 April and later sank with the loss of 11 lives , leading to the worst oil spill in US h
istory . 2010-7-6
The Deepwater Horizon drill rig exploded in April , leaving a ruptured well leaking hundreds o
f thousands of barrels of oil into the Gulf of Mexico . 2010-6-27
cents Commission told Transocean should have shut well cents Inquiry on rig blast starts with 
effects of disaster Boats tackle the blaze on the Deepwater Horizon oil rig that exploded in A
pril . 2010-7-13
The big winner of last year 's SAFE award was Transocean , the owner of the Deepwater Horizon 
rig that exploded last month under BP 's management . 2010-5-3
Since the Deepwater Horizon rig exploded on April 20 , BP 's share price has fallen over 34 % 
, including a huge 13 % fall on Tuesday . 2010-6-2
Six months ago , the Deepwater Horizon rig exploded , causing a massive oil spill in the Gulf 
of Mexico . 2010-10-25
"""

def write_docs_to_file(fname, docs):
  with open(fname, 'wb') as writer:
    for idx,doc in enumerate(docs):
      #if idx % 1000 == 0:
      #  print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(#num_stories))

      # Look in the tokenized story dirs to find the .story file corresponding to this url

      # Get the strings to write to .bin file
      abstract = "".encode("utf8")
      article = "\n".join(doc).encode("utf8")

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

if __name__ == "__main__":
  write_docs_to_file("test_1.bin", [[l.rsplit(" ", 1)[0] for l in data.split("\n")]])
