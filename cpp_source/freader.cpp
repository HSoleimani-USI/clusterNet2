#include "freader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// Takes a filename and creates a hashtable char -> index
//
freader::freader( const std::string newFname )
    : _fname( newFname ), _file( _fname, std::ifstream::in ) {

    // initialize the map with all ascii chars
    for ( int c = 0; c < 128; c++ ) {
        _char2indexTable[c] = -1;
    }

    // std::string m =
    // "abcdefghijklmnopqrstuvwxyz!?<>,.;:'{}01234567890-_=+!@#$%^&*()~`| \n";
    // for ( auto& c : m ) {
    //    _char2indexTable[c] = -1;
    //}

    char character = _file.get();
    int position = 0;
    while ( _file.good() ) {
        character = tolower( character );
        if ( _char2indexTable[character] == -1 ) {
            _char2indexTable[character] = position;
            position++;
        }
        character = _file.get();
        
    }

    _file.close();
}

void freader::printMap() {
    std::ifstream ifs( _fname, std::ifstream::in );
    char character = ifs.get();
    while ( ifs.good() ) {
        character = tolower( character );
        std::cout << character << " : " << _char2indexTable[character]
                  << std::endl;
        character = ifs.get();
    }

    ifs.close();
}

std::vector<int> freader::getValues() {
    std::vector<int> values;
    std::ifstream ifs( _fname, std::ifstream::in );
    int index = 0;
    char character = ifs.get();
    while ( ifs.good() ) {
        character = tolower( character );

        index = _char2indexTable[character];
        values.push_back( index );

        character = ifs.get();
    }
    ifs.close();
    return values;
}

int& freader::operator[]( const char& k ) { return _char2indexTable[k]; }

const std::string freader::filename() const { return _fname; }

std::vector<std::vector<int>> freader::getMatrix(const int k)
{
    std::vector<int> indices = getValues();
    
    std::vector<std::vector<int> > matrix;

    for ( int i = 0; i <= indices.size() - k; i++ ) {
        std::vector<int> row;
        for ( int j = 0; j < k;  j++ ) {
            row.push_back(indices.at( i + j ));
        }
        matrix.push_back(row);
    }

    std::cout << "\nPrinting the matrix:" << std::endl;
    for (auto& row : matrix) {
        for(auto& a : row) {
            std::cout << a << ", ";
        }
        std::cout << std::endl;
    }

    return matrix;
}

int main( int argc, const char* argv[] ) {
    freader fr( "hanieh.txt" );

    fr.printMap();
    fr.getMatrix(3);

    return 0;
}
