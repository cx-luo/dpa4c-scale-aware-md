namespace deepmd_dpa4c { namespace chain_reclaimed_widen {
__attribute__((visibility("hidden"))) bool small_nvt_armed();
}}
// Host rendezvous for the audited same-stream forward -> fit -> node sequence.
namespace deepmd_dpa4c { namespace chain_fit_node {
namespace {
struct Pending {
  Arguments forward{};
  cudaStream_t stream=nullptr;
  FitPayload fit{};
  Arguments node{};
  bool armed=false,queued=false,pipeline=false,node_queued=false;
  int forward_windows=0;
  cudaEvent_t forward_ready[8]{};
  cudaEvent_t forward_complete=nullptr;
};
thread_local Pending pending;
bool enabled(){static const bool yes=[] {const char* p=std::getenv("DPA_FIT_NODE_FIT_NODE_FUSED");return p&&p[0]=='1';}();return yes;}
long max_nodes(){static const long v=[] {const char* p=std::getenv("DPA_WIDE_FIT_NODE_FIT_NODE_MAX_NODES");long n=p?std::atol(p):4096;return n>0?n:4096;}();return v;}
bool pipeline_enabled(){static const bool yes=[] {const char* p=std::getenv("DPA_CONSUMER_PIPELINE_PIPE");return p&&p[0]=='1';}();return yes;}
bool fit_node_pipeline_enabled(){static const bool yes=[] {const char* p=std::getenv("DPA_FIT_NODE_PIPELINE_FIT_NODE_PIPE");return p&&p[0]=='1';}();return yes;}
int mode(){static const int v=[] {const char* p=std::getenv("DPA_FIT_NODE_MODE");int n=p?std::atoi(p):3;return n>=0&&n<=3?n:3;}();return v;}
bool outlined_node_vjp_enabled(){static const bool yes=[] {
  const char* p=std::getenv("DPA_OUTLINED_NODE_VJP");return p&&p[0]=='1';
}();return yes;}
long outlined_node_vjp_min_nodes(){static const long v=[] {
  const char* p=std::getenv("DPA_OUTLINED_NODE_VJP_MIN_NODES");
  const long n=p?std::atol(p):16384;return n>0?n:16384;
}();return v;}
long outlined_node_vjp_max_nodes(){static const long v=[] {
  const char* p=std::getenv("DPA_OUTLINED_NODE_VJP_MAX_NODES");
  const long n=p?std::atol(p):65536;return n>0?n:65536;
}();return v;}
long outlined_node_vjp_max_density(){static const long v=[] {
  const char* p=std::getenv("DPA_OUTLINED_NODE_VJP_MAX_DENSITY");
  const long n=p?std::atol(p):127;return n>0?n:127;
}();return v;}
bool outlined_node_vjp_selected(const Arguments& a,const FitPayload& f){
  const long density=a.node_count>0?
      (a.edge_count+a.node_count-1)/a.node_count:0;
  return outlined_node_vjp_enabled()&&f.n>=outlined_node_vjp_min_nodes()&&
      f.n<=outlined_node_vjp_max_nodes()&&density<=outlined_node_vjp_max_density();
}
void bad_sequence(){std::fprintf(stderr,"FIT_NODE invalid same-stream fit/node rendezvous; refusing unsafe output\n");std::abort();}
}
void begin_forward(const Arguments& a,cudaStream_t stream){
  if(!enabled())return;
  if(pending.queued||pending.node_queued||pending.forward_windows)bad_sequence();
  cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream,&capture);
  // The first uncaptured model call allocates the side stream, events, force
  // buffer and split table.  Subsequent model-graph captures can then join the
  // existing side stream through the recorded event dependencies.
  const bool large=pipeline_enabled()&&
      capture!=cudaStreamCaptureStatusInvalidated&&a.node_count>=100000;
  const long fit_node_pipeline_density=a.node_count>0?
      (a.edge_count+a.node_count-1)/a.node_count:0;
  const int fit_node_pipeline_density_min=std::getenv("FIT_NODE_PIPELINE_MIN_DENSITY")?
      std::max(1,std::atoi(std::getenv("FIT_NODE_PIPELINE_MIN_DENSITY"))):128;
  const bool medium=fit_node_pipeline_enabled()&&pipeline_enabled()&&
      capture!=cudaStreamCaptureStatusInvalidated&&a.node_count>=4096&&
      a.node_count<100000&&fit_node_pipeline_density>=fit_node_pipeline_density_min;
  pending.pipeline=large||medium;
  pending.armed=a.node_count>0&&(a.node_count<max_nodes()||large)&&
      a.node_begin==0&&!a.has_spin&&a.lmax==2&&a.table_width==8&&
      a.radial_modes==0&&a.descriptor&&a.state_out;
  if(pending.armed){pending.forward=a;pending.stream=stream;}
}
bool wants_forward_pipeline(const Arguments& a,cudaStream_t stream){
  static const bool requested=[] {
    const char* p=std::getenv("DPA_FORWARD_PIPELINE_FORWARD_PIPE");return p&&p[0]=='1';
  }();
  static const long min_nodes=[] {
    const char* p=std::getenv("FORWARD_PIPELINE_MIN_NODES");return p?std::atol(p):0L;
  }();
  static const long max_nodes=[] {
    const char* p=std::getenv("FORWARD_PIPELINE_MAX_NODES");return p?std::atol(p):0L;
  }();
  static const long min_density=[] {
    const char* p=std::getenv("FORWARD_PIPELINE_MIN_DENSITY");return p?std::atol(p):0L;
  }();
  static const long max_density=[] {
    const char* p=std::getenv("FORWARD_PIPELINE_MAX_DENSITY");return p?std::atol(p):0L;
  }();
  const long density=a.node_count>0?
      (a.edge_count+a.node_count-1)/a.node_count:0;
  const bool selected=a.node_count>=min_nodes&&
      (max_nodes<=0||a.node_count<=max_nodes)&&density>=min_density&&
      (max_density<=0||density<=max_density);
  return requested&&selected&&pending.armed&&pending.pipeline&&
      pending.stream==stream&&a.node_count==pending.forward.node_count&&
      a.edge_count==pending.forward.edge_count;
}
void set_forward_pipeline(int windows,const cudaEvent_t* ready,
                          cudaEvent_t complete,cudaStream_t stream){
  if(!wants_forward_pipeline(pending.forward,stream)||windows<2||windows>8||
      !ready||!complete||pending.forward_windows)bad_sequence();
  pending.forward_windows=windows;
  for(int i=0;i<windows;++i)pending.forward_ready[i]=ready[i];
  pending.forward_complete=complete;
}
int forward_pipeline_windows(cudaStream_t stream){
  if(stream!=pending.stream)return 0;
  return pending.forward_windows;
}
void wait_forward_window(int window,cudaStream_t stream){
  if(stream!=pending.stream||window<0||window>=pending.forward_windows)
    bad_sequence();
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream,pending.forward_ready[window],0));
}
void wait_forward_complete(cudaStream_t stream){
  if(stream!=pending.stream||pending.forward_windows<=0||
      !pending.forward_complete)bad_sequence();
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream,pending.forward_complete,0));
}
void consume_forward_pipeline(cudaStream_t stream){
  if(stream!=pending.stream||pending.forward_windows<=0)bad_sequence();
  pending.forward_windows=0;
  pending.forward_complete=nullptr;
  for(auto& event:pending.forward_ready)event=nullptr;
}
bool queue_fit(const FitPayload& f,cudaStream_t stream){
  if(!enabled()||!pending.armed)return false;
  pending.armed=false;
  const auto& a=pending.forward;
  if(stream!=pending.stream||f.n!=a.node_count||f.x!=a.descriptor||f.types!=a.atype)return false;
  pending.fit=f;pending.queued=true;return true;
}
bool launch_node(const Arguments& a,cudaStream_t stream){
  if(!enabled()||!pending.queued)return false;
  const auto& before=pending.forward;const auto f=pending.fit;
  if(stream!=pending.stream||a.node_count!=f.n||a.node_begin!=0||a.descriptor_gradient!=f.x||a.state!=before.state_out||a.moment_gradient!=before.state_out||a.has_spin||a.lmax!=2||a.table_width!=8||a.radial_modes!=0)bad_sequence();
  if(pending.pipeline){
    pending.node=a;pending.node_queued=true;pending.queued=false;return true;
  }
  pending.queued=false;
  const int m=mode(),blocks=int((f.n+15)/16);
  const bool narrow=f.n>=4096;
  const bool outlined=narrow&&m==3&&outlined_node_vjp_selected(before,f);
  // Only skip unused type-embedding descriptor-gradient stores. The forward
  // still consumes all 70 features; node_vjp consumes exactly features 0..61.
  // Require the integration adapter's explicit NVT capability. Keep NPT/NVE,
  // unsupported modes and all medium/large kernels on their original routes.
  static const bool shared_l2_requested=[] {
    const char* value=std::getenv("DPA_SHARED_FITTING_WEIGHTS_SMALL_SHARED_L2");
    return value && value[0]=='1';
  }();
  static const bool physical_gradient_requested=[] {
    const char* value=std::getenv("DPA_FIT_SMALL_PHYSICAL_GRADIENT_ONLY");
    return value && value[0]=='1';
  }();
  if(physical_gradient_requested && shared_l2_requested && !narrow && m==3 &&
      deepmd_dpa4c::chain_reclaimed_widen::small_nvt_armed()) {
    k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72,true,true>
        <<<blocks,32,0,stream>>>(f,a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    static bool physical_shown=false;
    if(!physical_shown) {
      cudaFuncAttributes attr{};int resident=0;
      cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72,true,true>);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72,true,true>,32,0);
      std::fprintf(stderr,"PHYSICAL_SMALL_FITTING_SMALL_SHARED_L2_PHYSICAL_GRADIENT nodes=%ld regs=%d local=%zu shared=%zu resident=%d gradient_stride=72 physical_width=62 full_energy_width=70\n",f.n,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);
      physical_shown=true;
    }
    return true;
  }

  if(narrow){
    if(m==0)k3n::chain_fit_node::fitting_node_fused_kernel<false,true,62><<<blocks,32,0,stream>>>(f,a);
    else if(m==1)k3n::chain_fit_node::fitting_node_fused_kernel<true,true,62><<<blocks,32,0,stream>>>(f,a);
    else if(m==2)k3n::chain_fit_node::fitting_node_fused_kernel<false,false,62><<<blocks,32,0,stream>>>(f,a);
    else if(outlined) {
      k3n::chain_fit_node::fitting_node_outlined_kernel<<<blocks,32,0,stream>>>(f,a);
      k3n::chain_fit_node::outlined_node_vjp_kernel<<<int((f.n+127)/128),128,0,stream>>>(a);
    }
    else k3n::chain_fit_node::fitting_node_fused_kernel<true,false,62><<<blocks,32,0,stream>>>(f,a);
  }else{
    if(m==0)k3n::chain_fit_node::fitting_node_fused_kernel<false,true,72><<<blocks,32,0,stream>>>(f,a);
    else if(m==1)k3n::chain_fit_node::fitting_node_fused_kernel<true,true,72><<<blocks,32,0,stream>>>(f,a);
    else if(m==2)k3n::chain_fit_node::fitting_node_fused_kernel<false,false,72><<<blocks,32,0,stream>>>(f,a);
    else k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72><<<blocks,32,0,stream>>>(f,a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  static bool shown=false;
  if(!shown){cudaFuncAttributes attr{};int resident=0;
    if(narrow){
      if(m==0){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<false,true,62>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<false,true,62>,32,0);}
      if(m==1){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<true,true,62>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<true,true,62>,32,0);}
      if(m==2){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<false,false,62>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<false,false,62>,32,0);}
      if(m==3&&outlined){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_outlined_kernel);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_outlined_kernel,32,0);
        cudaFuncAttributes vjp_attr{};int vjp_resident=0;
        cudaFuncGetAttributes(&vjp_attr,k3n::chain_fit_node::outlined_node_vjp_kernel);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&vjp_resident,k3n::chain_fit_node::outlined_node_vjp_kernel,128,0);
        const long density=before.node_count>0?(before.edge_count+before.node_count-1)/before.node_count:0;
        std::fprintf(stderr,"OUTLINED_NODE_VJP nodes=%ld density=%ld regs=%d local=%zu shared=%zu resident128=%d\n",f.n,density,vjp_attr.numRegs,vjp_attr.localSizeBytes,vjp_attr.sharedSizeBytes,vjp_resident);}
      if(m==3&&!outlined){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,62>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,62>,32,0);}
    }else{
      if(m==0){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<false,true,72>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<false,true,72>,32,0);}
      if(m==1){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<true,true,72>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<true,true,72>,32,0);}
      if(m==2){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<false,false,72>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<false,false,72>,32,0);}
      if(m==3){cudaFuncGetAttributes(&attr,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k3n::chain_fit_node::fitting_node_fused_kernel<true,false,72>,32,0);}
    }
    std::fprintf(stderr,"FIT_NODE_FIT_NODE_FUSED nodes=%ld mode=%d gradient_width=%d outlined=%d regs=%d local=%zu shared=%zu resident=%d\n",f.n,m,narrow?62:72,int(outlined),attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);shown=true;
  }
  return true;
}

bool take_large_pipeline(FitPayload& fit,Arguments& node,cudaStream_t stream){
  if(!enabled()||!pipeline_enabled()||!pending.node_queued)return false;
  if(stream!=pending.stream)bad_sequence();
  fit=pending.fit;node=pending.node;
  pending.node_queued=false;pending.pipeline=false;pending.armed=false;
  return true;
}

void launch_fused_range(const FitPayload& input,const Arguments& input_node,
                        long begin,long count,cudaStream_t stream){
  if(begin<0||count<=0||begin+count>input.n||mode()!=3)std::abort();
  FitPayload f=input;
  f.n=count;
  f.x+=begin*70;
  f.types+=begin;
  f.seed+=begin;
  f.energy+=begin;
  Arguments a=input_node;
  a.node_begin=0;
  a.node_count=count;
  a.state+=begin*42;
  a.descriptor_gradient+=begin*70;
  a.moment_gradient+=begin*42;
  const int blocks=int((count+15)/16);
  k3n::chain_fit_node::fitting_node_fused_kernel<true,false,62>
      <<<blocks,32,0,stream>>>(f,a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}}
