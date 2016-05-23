#ifndef FREADER_INCLUDED
#define FREADER_INCLUDED

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

class freader {
public:
	std::vector<int> generateMatrix(int ifs, int k);

    freader( const std::string );

    void buildMap();
    void printMap();
    std::vector<std::vector<int>> getMatrix(const int k);

    int& operator[]( const char& k );
    std::vector<int> getValues();
    const std::string filename() const;

private:
    const std::string _fname;
    std::ifstream _file;
    std::unordered_map<char, int> _char2indexTable;
};

#endif
