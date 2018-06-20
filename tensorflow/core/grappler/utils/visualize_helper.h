//
// Created by a1 on 6/8/18.
//

#ifndef  THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_UTILS_VISUALIZE_HELPER_H
#define  THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_UTILS_VISUALIZE_HELPER_H


#include <string>

namespace tensorflow {
namespace grappler{


class VisualizeHelper {
public:
  void PrintGraph(GraphDef * g, std::string & content){
    std::cout<<__FUNCTION__<<std::endl;
    content = "digraph mygraph {\n";
    for(auto n : g->node()){
      PrintNode(g, n, content);
    }
    content += "}";
  }

  void PrintNode(GraphDef * g, const NodeDef& node, std::string & content){
    std::string dest = "\"" + node.name() + ":" + node.op() + "\"";
    for(auto input : node.input()){
      std::string line;
      if(input[0] == '^'){
        std::string input_node_name = input.substr(1);

        line = "\"" + input_node_name;
        for(auto n : g->node()){
          if(input_node_name == n.name()){
            line += ":" + n.op() + "\"";
            break;
          }
        }
        line += "->" + dest;
        line += "[style=dashed]";
      }else{
        std::string input_node_name = input;

        line = "\"" + input_node_name;
        for(auto n : g->node()){
          if(input_node_name == n.name()){
            line += ":" + n.op() + "\"";
            break;
          }
        }
        line += "->" + dest;
      }
      content += line + ";\n";
    }
  }
};


}  // namespace grappler
}  // namespace tensorflow

#endif // THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_UTILS_VISUALIZE_HELPER_H
