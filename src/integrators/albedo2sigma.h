// this head file is used for skinpath integrator, which provides a albedo value and need the responding sigma value
// ABORT!!!!

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "geometry.h"


namespace pbrt {
// C++ does not have split function, so we must write one
void SplitString(const std::string& s, std::vector<std::string>& v,
                 const std::string& c);

// read LUT value from file
void ReadLUT(std::vector<Vector3f>& colors, std::vector<Vector3f>& sigmaaOutter,
             std::vector<Vector3f>& sigmaaInner);

}  // namespace pbrt


