#define RT_FUNCTION __noinline__ __device__

RT_FUNCTION float3 test()
{
    return make_float3(1,0,0);
}
