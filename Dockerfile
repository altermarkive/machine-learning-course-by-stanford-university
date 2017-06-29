FROM python:3.6.0

COPY requirements.1.txt /tmp/requirements.1.txt
COPY requirements.2.txt /tmp/requirements.2.txt

RUN pip3 install -r /tmp/requirements.1.txt && \
    pip3 install -r /tmp/requirements.2.txt

# docker build -t alternative .
# octave --eval warmUpExercise
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex1/ex1 alternative python -c 'from warmUpExercise import *; print(warmUpExercise())'
# octave --eval ex1
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex1/ex1 alternative python -c 'from ex1 import *; ex1()'
# octave --eval ex1_multi
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex1/ex1 alternative python -c 'from ex1_multi import *; ex1_multi()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex1/ex1 alternative python -c 'from submit import *; submit()'
# octave --eval ex2
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex2/ex2 alternative python -c 'from ex2 import *; ex2()'
# octave --eval ex2_reg
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex2/ex2 alternative python -c 'from ex2_reg import *; ex2_reg()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex2/ex2 alternative python -c 'from submit import *; submit()'
# octave --eval ex3
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex3/ex3 alternative python -c 'from ex3 import *; ex3()'
# octave --eval ex3_nn
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex3/ex3 alternative python -c 'from ex3_nn import *; ex3_nn()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex3/ex3 alternative python -c 'from submit import *; submit()'
# octave --eval ex4
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex4/ex4 alternative python -c 'from ex4 import *; ex4()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex4/ex4 alternative python -c 'from submit import *; submit()'
# octave --eval ex5
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex5/ex5 alternative python -c 'from ex5 import *; ex5()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex5/ex5 alternative python -c 'from submit import *; submit()'
# octave --eval ex6
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex6/ex6 alternative python -c 'from ex6 import *; ex6()'
# octave --eval ex6_spam
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex6/ex6 alternative python -c 'from ex6_spam import *; ex6_spam()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex6/ex6 alternative python -c 'from submit import *; submit()'
# octave --eval ex7
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex7/ex7 alternative python -c 'from ex7 import *; ex7()'
# octave --eval ex7_pca
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex7/ex7 alternative python -c 'from ex7_pca import *; ex7_pca()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex7/ex7 alternative python -c 'from submit import *; submit()'
# octave --eval ex8
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex8/ex8 alternative python -c 'from ex8 import *; ex8()'
# octave --eval ex8_cofi
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex8/ex8 alternative python -c 'from ex8_cofi import *; ex8_cofi()'
# octave --eval submit
# docker run -it --rm -v $PWD:/code -w /code/machine-learning-ex8/ex8 alternative python -c 'from submit import *; submit()'
