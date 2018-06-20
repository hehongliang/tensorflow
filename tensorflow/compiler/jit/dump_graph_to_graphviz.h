//
// Created by a1 on 6/11/18.
//

#ifndef TENSORFLOW_COMPILER_JIT_DUMP_GRAPH_TO_GRAPHVIZ_H
#define TENSORFLOW_COMPILER_JIT_DUMP_GRAPH_TO_GRAPHVIZ_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorflow {

class GraphDef;
class NodeDef;
class Graph;
class FunctionDef;

class DumpGraphToGraphviz {
public:
  void PrintGraph(Graph * g, std::unordered_map<std::string, std::string> * cluster, std::string & content);
  void PrintGraph(GraphDef * g, std::string & content);


private:

  class State {
  public:
    void Reset(GraphDef * g, std::unordered_map<std::string, std::string> * cluster);

    std::string Run();

  private:
    void AddGVNode(const std::string& name, const std::string& label);
    void AddGVEdge(const std::string& src, const std::string& dest, bool control_flow);
    std::string * GetGraphEdgeContainer(const std::string& src, const std::string& dest);
    std::string GetNodeLabel(const NodeDef& node);
    std::string ToString();

  private:
    GraphDef * g_ = nullptr;
    std::unordered_map<std::string, std::string> * cluster_ = nullptr;

    std::unordered_map<std::string, std::string> cluster_edge_content_container_;
    std::string root_edge_content_container_;
    std::string root_node_content_container_;
  };

  State state_;

};

class DumpFunctionToGraphviz {
public:
  void PrintGraph(FunctionDef * f, std::string & content);

private:
  void AddGVNode(const std::string& name, const std::string& label, std::string & content);
  void AddGVEdge(const std::string& src, const std::string& dest, bool control_flow, std::string & content);
};



}



#endif //TENSORFLOW_COMPILER_JIT_DUMP_GRAPH_TO_GRAPHVIZ_H
