#include "freader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <Matrix.h>
#include <string.h>

using std::endl;
using std::cout;

// Takes a filename and creates a hashtable char -> index
freader::freader( const std::string newFname, ClusterNet *acc)
    : _fname( newFname ), _file( _fname, std::ifstream::in ) {

	_acc = acc;
	_current_offset = 0;
    // initialize the map with all ascii chars
    for ( int c = 0; c < 128; c++ ) {
        _char2indexTable[c] = -1.0f;
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
            _char2indexTable[character] = (float)position;
            position++;
        }
        character = _file.get();
        
    }

    _file.close();
}


std::string freader::read_chunk(std::string path, int offset, int size)
{
	std::ifstream t(path);
	t.seekg(0, std::ios::end);
	size_t fullsize = t.tellg();
	if(size > fullsize - offset){ size = fullsize - offset; }
	std::string buffer(size, ' ');
	t.seekg(offset, std::ios::cur);
	t.read(&buffer[0], size);
	_current_offset = 0;
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

std::vector<float> freader::getValues() {
    std::vector<float> values;
    std::ifstream ifs( _fname, std::ifstream::in );
    float index = 0.0f;
    char character = ifs.get();
    cout << character << ifs.good() << endl;
    while ( ifs.good() ) {
        character = tolower( character );
	cout << character << endl;

        index = _char2indexTable[character];
        values.push_back( index );

        character = ifs.get();
    }
    ifs.close();
    return values;
}

float& freader::operator[]( const char& k ) { return _char2indexTable[k]; }

const std::string freader::filename() const { return _fname; }

Matrix<float> *freader::getMatrix(const int sequence_length, const int batch_size)
{
    std::vector<float> indices = getValues();

    Matrix<float> *ret = _acc->OPS->zeros(batch_size, sequence_length);
    _acc->OPS->to_host(ret,ret->data);


    for(int i = 0; i < sequence_length; i++)
	cout << indices[i] << " ";
    cout << endl;

    cout << "pre memcpy" << endl;
    for ( int i = 0; i <= sequence_length; i++ )
    {
	cout << i << endl;
    	memcpy(&(ret->data[sequence_length*i]),&(indices[(_current_offset+i)]), sequence_length);
    }

    _current_offset += batch_size;
    return ret;
}
