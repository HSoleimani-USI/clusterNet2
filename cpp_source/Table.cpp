/*
 * Table.cpp
 *
 *  Created on: Feb 17, 2016
 *      Author: tim
 */

#include "Table.h"
#include "json.hpp"
#include <vector>
#include <regex>

// for convenience
using json = nlohmann::json;

using std::cout;
using std::endl;

Table::Table(std::string name, std::string path)
{
	mypath = path+"/"+name;
	myname = name;


	leveldb::Options options;
	options.create_if_missing = true;
	leveldb::Status status = leveldb::DB::Open(options, path + "/" + name, &db);
}

/*
 *
j.is_null();
j.is_boolean();
j.is_number();
j.is_object();
j.is_array();
j.is_string();
 *
 */

template std::map<std::string,int> Table::get_dictionary(std::string key);
template <typename K,typename V> std::map<K,V> Table::get_dictionary(std::string key)
{
	std::map<K,V> ret;

	std::string value;
    leveldb::Status s = db->Get(leveldb::ReadOptions(), key, &value);
    auto obj = json::parse(value);
    assert(obj.is_object());
	for (json::iterator it = obj.begin(); it != obj.end(); ++it)
	{
	  ret[(K)it.key()] = (V)it.value();
	}

	return ret;
}


bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

/*
template Matrix<float> *Table::get_data(std::string key);
template Matrix<int> *Table::get_data(std::string key);
template <typename T> Matrix<T>*Table::get_data(std::string key)
{
	replace(key, "/","_");
    std::string path = mypath + "/hdf5/"+key + ".hdf5";
    Matrix<T> *ret =read_hdf5<T>(path.c_str());

	return ret;
}
*/
