// this head file is used for skinpath integrator, which provides a albedo value and need the responding sigma value

// ABORT!!!!

#include "albedo2sigma.h"

namespace pbrt {
// C++ does not have split function, so we must write one
void SplitString(const std::string& s, std::vector<std::string>& v,
                 const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) v.push_back(s.substr(pos1));
}

// read LUT value from file
void ReadLUT(std::vector<Vector3f> &colors,
             std::vector<Vector3f> &sigmaaOutter,
             std::vector<Vector3f> &sigmaaInner) {
    std::string LUTFilePath = "lut_test_twolayer.txt";
    std::fstream LUTFile;
    LUTFile.open(LUTFilePath, std::ios::in);
    if (LUTFile.is_open()) {
        std::string temp;
        std::vector<std::string> v;
        std::vector<std::string> colorVector;
        std::vector<std::string> sigmaaOutterVector;
        std::vector<std::string> sigmaaInnerVector;
        Vector3f color;
        Vector3f sigmaOutter;
        Vector3f sigmaInner;
        while (std::getline(LUTFile,temp)) {
			// process lines here
            SplitString(temp, v, " ");
            SplitString(v[0], colorVector, ",");
            SplitString(v[1], sigmaaOutterVector, ",");
            SplitString(v[2], sigmaaInnerVector, ",");
            color = {float(atof(colorVector[0].c_str())),
                     float(atof(colorVector[1].c_str())),
                     float(atof(colorVector[2].c_str()))};
            sigmaOutter = {float(atof(sigmaaOutterVector[0].c_str())),
                           float(atof(sigmaaOutterVector[1].c_str())),
                           float(atof(sigmaaOutterVector[2].c_str()))};
            sigmaInner = {float(atof(sigmaaInnerVector[0].c_str())),
                          float(atof(sigmaaInnerVector[1].c_str())),
                          float(atof(sigmaaInnerVector[2].c_str()))};
            colors.emplace_back(color);
            sigmaaOutter.emplace_back(sigmaOutter);
            sigmaaInner.emplace_back(sigmaInner);
            v.clear();
            colorVector.clear();
            sigmaaOutterVector.clear();
            sigmaaInnerVector.clear();
        }
        LUTFile.close();
	}
}

void test() { std::cout << "test" << std::endl; }


}  // namespace pbrt


