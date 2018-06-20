//
// Created by a1 on 6/11/18.
//

#include <sstream>

#include "tensorflow/compiler/jit/dump_graph_to_graphviz.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/graph/graph.h"


namespace tensorflow {


void DumpGraphToGraphviz::State::Reset(GraphDef * g, std::unordered_map<std::string, std::string> * cluster){
  g_ = g;
  cluster_ = cluster;
  cluster_edge_content_container_.clear();
  root_edge_content_container_.clear();
  root_node_content_container_.clear();
}

std::string * DumpGraphToGraphviz::State::GetGraphEdgeContainer(const std::string& src, const std::string& dest){
  if(!cluster_){
    return &root_edge_content_container_;
  }

  if(cluster_->find(src) == cluster_->end() || cluster_->find(dest) == cluster_->end() ){
    return &root_edge_content_container_;
  }

  if((*cluster_)[src] != (*cluster_)[dest]){
    return &root_edge_content_container_;
  }

  return &cluster_edge_content_container_[(*cluster_)[dest]];
}


void DumpGraphToGraphviz::State::AddGVNode(const std::string& name, const std::string& label){
  root_node_content_container_ += name + "[label=" + label + "];\n";
}

void DumpGraphToGraphviz::State::AddGVEdge(const std::string& src, const std::string& dest, bool control_flow){
  auto pcontainer = GetGraphEdgeContainer(src, dest);
  *pcontainer += src + "->" + dest;
  if(control_flow){
    *pcontainer += "[style=dashed];\n";
  }else{
    *pcontainer += ";\n";
  }
}

std::string DumpGraphToGraphviz::State::Run(){
  for(auto node : g_->node()){
    AddGVNode(node.name(), GetNodeLabel(node));
    for(auto input : node.input()){
      if(input[0] == '^'){
        std::string src_node_name = input.substr(1);
        AddGVEdge(src_node_name, node.name(), true);
      }else{
        AddGVEdge(input, node.name(), false);
      }
    }
  }

  return ToString();
}

std::string DumpGraphToGraphviz::State::GetNodeLabel(const NodeDef& node){
  std::string label = "\"" + node.name() + ":" + node.op() + "\"";
  return label;
}

std::string DumpGraphToGraphviz::State::ToString(){
  std::string content = "digraph MyGraph { \n" + root_node_content_container_;
  for(auto it = cluster_edge_content_container_.begin(); it != cluster_edge_content_container_.end(); it++){
    content += "subgraph " + it->first + "{\n" + it->second + "}\n";
  }
  content += root_edge_content_container_ + "}\n";
  return content;
}



void DumpGraphToGraphviz::PrintGraph(Graph * g, std::unordered_map<std::string, std::string> * cluster, std::string & content){
  GraphDef def;
  g->ToGraphDef(&def);
  state_.Reset(&def, cluster);
  content = state_.Run();
}


void DumpGraphToGraphviz::PrintGraph(GraphDef * g, std::string & content){
  state_.Reset(g, nullptr);
  content = state_.Run();
}


void DumpFunctionToGraphviz::PrintGraph(FunctionDef * f, std::string & content) {
  std::string node_string;
  std::string edge_string;
  for(auto node : f->node_def()){
    AddGVNode(node.name(), "\"" + node.name() + ":" + node.op() + "\"", node_string);
    for(auto input : node.input()){
      if(input[0] == '^'){
        std::string src_node_name = input.substr(1);
        AddGVEdge(src_node_name, node.name(), true, edge_string);
      }else{
        AddGVEdge(input, node.name(), false, edge_string);
      }
    }
  }
  content = "digraph MyGraph { \n" + node_string + edge_string + "}\n";
}

void DumpFunctionToGraphviz::AddGVNode(const std::string& name, const std::string& label, std::string & content){
  content += name + "[label=" + label + "];\n";
}

void DumpFunctionToGraphviz::AddGVEdge(const std::string& src, const std::string& dest, bool control_flow, std::string & content){
  content += src + "->" + dest;
  if(control_flow){
    content += "[style=dashed];\n";
  }else{
    content += ";\n";
  }
}




}
