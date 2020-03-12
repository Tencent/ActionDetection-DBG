TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}
all:
	nvcc -std=c++11 -O2 -c -o src/prop_tcfg_op.cu.o src/prop_tcfg_op.cu.cc \
			$(TF_CFLAGS) $(LDFLAGS) -I/usr/local -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
			-DNDEBUG --expt-relaxed-constexpr -w
	g++ $(CFLAGS) -o prop_tcfg.so src/prop_tcfg_op.cc src/prop_tcfg_op.cu.o \
			$(LDFLAGS) -L/usr/local/cuda/lib64 -D GOOGLE_CUDA=1 \
			-I/usr/local -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include \
			-L/usr/local/cuda/targets/x86_64-linux/lib -lcudart 
