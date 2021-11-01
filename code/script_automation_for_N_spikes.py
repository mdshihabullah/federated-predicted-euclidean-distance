import subprocess
import sys

script_name = str(sys.argv[1])
output_prefix = 'out'
n_iter = 26

for i in range(n_iter):
    output_file = '' + script_name + output_prefix + '_' + str(i) + '.txt'
    sys.stdout = open(output_file, 'w')
    subprocess.call(['python', script_name+' '+i], stdout=sys.stdout, stderr=subprocess.STDOUT)