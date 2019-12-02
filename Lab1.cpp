#include <map>
#include <locale.h>
#include <fstream>
#include <iostream>
#include "Lab1.h"
#include "SSLConnection\SSLConnection.h"
#include "JSONParser\JsonDoc.h"
#include <algorithm>
#include "StemmerParser\StremmerPorter.h"
#include "NeuralNetwork\NeuralNetwork.h"
#include "nlohmann\json.hpp"
#include "Data Parser\Data.h"
#include "NeuralNetwork\Misc.h"

using namespace std;
using json = nlohmann::json;

string UnicodeToUTF8(char* data)
{
	string res;
	while (*data)
	{
		if (*data == '\\' && *(data + 1) == 'u')
		{
			data += 2;
			string utf16;
			while (*data != '\\' && *data != ' ' && *data != ',' && *data != '.')
			{
				utf16.push_back(*data);
				data++;
			}

			int wchar = strtol(utf16.data(), 0, 16);
			unsigned char new_char = wchar - 848;

			if (wchar == 1105) 
			{
				new_char += 73;
			}
			else if (wchar == 1025) 
			{
				new_char += 9;
			}
			res += new_char;
		}
		else
		{
			res += *data;
			data++;
		}
	}
	return res;
}

void clearFrom(char* string, char delimiter)
{
	char* pData = string;

	while (*string)
	{
		if (*string == delimiter)
		{
			char* next_char = string + 1, * current_char = string;
			while (*next_char)
			{
				*current_char = *next_char;
				current_char++;
				next_char++;
			}
			*current_char = '\0';
		}
		string++;
	}
}

ANNConfig buildConfigs(json configObject)
{
	ANNConfig configs;

	vector<int> topology = configObject["topology"];

	string trainingFile = configObject["trainingData"];
	string labelsFile = configObject["labelData"];
	string weightsFile = configObject["weightsFile"];

	configs.topology = topology;

	configs.bias = (int)configObject["bias"];
	configs.learningRate = (double)configObject["learningRate"];
	configs.momentum = (double)configObject["momentum"];

	configs.epoch = (int)configObject["epoch"];

	configs.hActivation = (ANN_ACTIVATION)configObject["hActivation"];
	configs.oActivation = (ANN_ACTIVATION)configObject["oActivation"];

	configs.trainingFile = trainingFile;
	configs.labelsFile = labelsFile;
	configs.weightsFile = weightsFile;

	return configs;
}

int main()
{
	setlocale(LC_NUMERIC, "C");
	srand(time(0));

	Data featureVectors;

	/*featureVectors.loadDictionaryData("summaries");
	featureVectors.stremmerParse();

	featureVectors.eraseSymbols(",.!?;()[]{}:/%+-\"\'0123456789_*~=`ЂїЧ∞");

	featureVectors.splitData();

	featureVectors.createWordsDictionary("summaries_new");
	featureVectors.vectorizeData();
	featureVectors.saveTraininigData("trainingData.csv");
	featureVectors.saveTestData("testData.csv");*/

	//featureVectors.loadLabels("summaries");

	vector<vector<double>> labels = featureVectors.labels;
	ifstream configFile("Config.txt", ios_base::binary);

	string jsonStr;
	jsonStr.assign(istreambuf_iterator<char>(configFile), istreambuf_iterator<char>());

	configFile.close();
	json configObject = json::parse(jsonStr);

	NeuralNetwork nn(buildConfigs(configObject));

	vector<vector<double>> trainingData = utils::Misc::fetchData("testData.csv");
 
	int SIZE = 500;

	for (int i = 0; i < trainingData.size(); i++)
	{
		if (trainingData[i].size() < SIZE)
		{
			int diff = SIZE - trainingData[i].size();
			for (int j = 0; j < diff; j++)
			{
				trainingData[i].push_back(0.0);
			}
		}
		else if (trainingData[i].size() > SIZE)
		{
			trainingData[i].resize(SIZE);
		}
	}

	nn.loadWeights("weights.json");

	for (int i = 0; i < trainingData.size(); i++)
	{
		nn.setCurrentInput(trainingData[i]);

		nn.feedForward();
		nn.setErrors();

		vector<Neuron*> nrns = nn.layers.at(nn.layers.size() - 1)->getNeurons();
		cout << nrns[0]->getActivatedVal() << ' '  << nrns[1]->getActivatedVal() << '\n';
	}

	// TRAIN PART

	/*for (int i = 0; i < nn.config.epoch; i++)
	{ 
		for (int j = 0; j < trainingData.size(); j++)
		{
			nn.train(trainingData[j], labels[j], nn.config.bias, nn.config.learningRate, nn.config.momentum);
			cout << nn.error << '\n';
		}

	}

	nn.saveWeights("weights.json");*/

	SSLConnection* connection = new SSLConnection();
	string data;

	JsonDoc json;

	//for (int i = 1; i < 2; i++) // получение данных  авто с onliner 
	//{
	//	string message = "GET /sdapi/ab.api/search/vehicles?page=";
	//	message += std::to_string(i) + "&extended=true";
	//	message += " HTTP/1.1\r\nAccept: application/json, text/javascript, */*\r\nHost: ab.onliner.by\r\n\r\n";

	//	connection->connectToEncryptHost(443, "ab.onliner.by");
	//	connection->handShake("ab.onliner.by");

	//	connection->sendMessage(message.c_str());
	//	data = connection->recieveMessage();

	//	string data_new(data);
	//	data_new.erase(std::remove(data_new.begin(), data_new.end(), '"'), data_new.end());

	//	connection->disconnect();

	//	char* json_string = nullptr;
	//	char* headers = strtok_s((char*)data_new.c_str(), "{", &json_string); // extract json part

	//	JsonObject* json_data = new JsonObject;
	//	json.fromJson(&json_string, json_data);

	//	JsonArray* adverts = (JsonArray*)json_data->data["adverts"];

	//	for (int j = 0; j < adverts->data.size(); j++)
	//	{
	//		JsonObject* objects_to_print = (JsonObject*)adverts->data[j];
	//		wcout << j << '\n';

	//		char* url = (char*)((string*)objects_to_print->data["html_url"])->c_str();
	//		char* new_url = url + 25; // pass through ab.onliner.by

	//		char* context = nullptr;
	//		char* old = strtok_s(new_url, "/", &context);
	//		old = strtok_s(nullptr, "/", &context);

	//		message = "GET /sdapi/ab.api/vehicles/";
	//		message += context;
	//		message += " HTTP/1.1\r\nAccept: application/json, text/javascript, */*\r\nHost: ab.onliner.by\r\n\r\n";

	//		connection->connectToEncryptHost(443, "ab.onliner.by");
	//		connection->handShake("ab.onliner.by");

	//		connection->sendMessage(message.c_str());
	//		data = connection->recieveMessage();

	//		string new_data = UnicodeToUTF8((char*)data.data());
	//		clearFrom((char*)new_data.data(), '"');

	//		connection->disconnect();

	//		char* json_string = nullptr;
	//		char* headers = strtok_s((char*)new_data.c_str(), "{", &json_string); // extract json part

	//		JsonObject* json_data = new JsonObject;
	//		json.fromJson(&json_string, json_data);

	//		ofstream data_file("summaries", ios_base::binary | ios_base::app);
	//		data_file << ((string*)json_data->data["description"])->c_str() << "\r\n";
	//		data_file.close();

	//		delete json_data;
	//		cout << "Wrote to file.\n";
	//	}

	//	delete json_data;
	//}
	//delete connection;

	cout << "ok!!!";

	getchar();
	return 0;
}

