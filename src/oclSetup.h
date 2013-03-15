#include <Cl/opencl.h>

char* readSourceFile(const char*);
void oclSetup(cl_platform_id*, cl_device_id*, cl_context*,
              cl_command_queue*, cl_program*, char*);
