#include <stdio.h>
#include <stdlib.h>
#include "httplib.h"
#include <string>

void doGetHi(const httplib::Request& req, httplib::Response& res, const httplib::ContentReader& content_reader)
{
    std::string body;
    content_reader([&](const char* data, size_t len){
        body.append(data, len);
        return true;
    });
    res.set_content(body, "text/plain");
}

int main(int argc, char *argv[])
{
    httplib::Server server;
    server.Post("/hi", doGetHi);
    server.listen("0.0.0.0", 8081);
    return 0;
}

// g++ server.cpp -o server -pthread
// curl http://localhost:8081/hi -d message
