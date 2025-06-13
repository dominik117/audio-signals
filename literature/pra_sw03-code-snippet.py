# globals().clear()

import numpy as np
import hmm


# -----------------------------------------------------------------------------
# basic dictionaries to convert characters to morse code and its 'binary' 
# representation
# -----------------------------------------------------------------------------

# morse coding
alphabet = list('abcdefghijklmnopqrstuvwxyz')
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', 
          '.---', '-.-', '.-..', '--', '-.','---', '.--.', '--.-', 
          '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..']

morse_dict = dict(zip(alphabet, values))
ascii_dict = dict(map(reversed, morse_dict.items())) # inverse mapping

# convert text to morse code
def morse_encode(text):
    return ' '.join([''.join( ('['+morse_dict.get(i, '')+']') for i in text)])
 
# encode morse symbols to observable values
code = list('.-[]')  # dot, dash and start/stop
observable = [0,1,2,3]

observable_dict = dict(zip(code, observable))

# convert morse code to observable data stream
def observable_encode(morse_code):
    return [observable_dict.get(i, []) for i in morse_code]
 
# -----------------------------------------------------------------------------

# example encodings 
morse_encode('j')
observable_encode(morse_encode('j'))
morse_encode('hslu')
observable_encode(morse_encode('hslu'))

# -----------------------------------------------------------------------------
# train a model for each character of the alphabet
# -----------------------------------------------------------------------------

sequence = 'x'      # training sequence

AA=[]
BB=[]
for i in range(len(alphabet)):
    print('Train : '+alphabet[i])
    s = sequence.replace('x',alphabet[i])    
    O = np.array(observable_encode(morse_encode(s)))
    
    A,B,pi = hmm.new_model(num_states = 10)
    A, B = hmm.baum_welch(O, A, B, pi, n_iter=50)
    AA.append(A)
    BB.append(B)
    
# -----------------------------------------------------------------------------
# test all models for all characters
# -----------------------------------------------------------------------------

for c in range(len(alphabet)):
    O = np.array(observable_encode(morse_encode(alphabet[c])))
    r = np.zeros(len(alphabet))
    for i in range(len(alphabet)):      
        a = hmm.forward(O, AA[i], BB[i], pi)
        r[i] = sum(a[O.shape[0]-1,:])

        # alternate ranking by viterbi sequence
        #q = hmm.viterbi(O, AA[i], BB[i], pi)    # most probable state sequence
        #p = pi[q[0]]
        #for j in range(q.shape[0]-1):           # probability of sequence q 
        #    p = p * AA[i][q[j],q[j+1]]
        #r[i] = p
        
    print('Test :',alphabet[c],'->',alphabet[np.argmax(r)],'' if c == np.argmax(r) else ' <= FAIL')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


