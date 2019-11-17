#ifndef OP_PROP_TCFG_H_
#define OP_PROP_TCFG_H_

//#define FEATURE_NUM 32
namespace tensorflow {

	namespace functor {
		template <typename Device, typename T>
		struct PropTcfgFunctor {
			void operator()(const Device& d,
				int batch, int tscale, int in_channel,
				const T* input, T* out, int mode,
				int start_num, int center_num, int end_num);
		};

		template <typename Device, typename T>
		struct PropTcfgGradFunctor {
			void operator()(const Device& d,
				int batch, int tscale, int in_channel,
				const T* output_grad, T* input_grad, int mode,
				int start_num, int center_num, int end_num);
		};
	}	// namespace functor

}	// namespace tensorflow
#endif	// OP_PROP_TCFG_H_
