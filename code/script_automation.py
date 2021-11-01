import subprocess
import sys

script_name = str(sys.argv[1])
n_iter = 1

for i in range(n_iter):
    output_file = '' + script_name + '' + '_' + str(i) + '.txt'
    sys.stdout = open(output_file, 'w')
    subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)