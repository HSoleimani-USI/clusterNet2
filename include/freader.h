#ifndef FREADER_INCLUDED
#define FREADER_INCLUDED

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <Matrix.h>
#include <ClusterNet.h>
#include <string.h>

class freader {
public:
	std::vector<int> generateMatrix(int ifs, int k);

    freader( const std::string, ClusterNet *acc );

    void buildMap();
    void printMap();
    Matrix<float> *getMatrix(const int sequence_length, const int batch_size);

    float& operator[]( const char& k );
    std::vector<float> getValues();
    const std::string filename() const;
    std::string read_chunk(std::string path, int offset, int size);

private:
    const std::string _fname;
    std::ifstream _file;
    std::unordered_map<char, float> _char2indexTable;
    int _current_offset;
    ClusterNet *_acc;
};

#endif
