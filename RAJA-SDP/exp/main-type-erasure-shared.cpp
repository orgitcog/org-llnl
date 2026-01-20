// g++ --std=c++11 main-tye_erasure.cpp
#include <iostream>
#include <memory>
class GPU
{
  public:
    void print_name() const { std::cout << "I am GPU " << value << std::endl; }
    void do_GPU() { std::cout << "Do GPU specific member function" << std::endl; }
    void set_value(int i) { value = i; }
  private:
    int value = 1;
};

class Host
{
  public:
    void print_name() const { std::cout << "I am Host " << value << std::endl; }
    void do_Host() { std::cout << "Do Host specific member funtion." << std::endl; }
    void set_value(int i) { value = i; }
  private:
    int value = 1;
};


class Context
{
  public:

    template<typename T>
    Context(T&& value){ m_value.reset(new ContextModel<T>(value)); }

    void print_name() const { m_value->print_name(); }
    void set_value(int i) const { m_value->set_value(i); }

    template<typename T>
    T get() { 
      auto result = dynamic_cast<ContextModel<T>*>(m_value.get()); 
      if (result == nullptr)
      {
        std::cout << "NULLPTR" << std::endl;
        std::exit(1);
      }
      return result->get();
    }

  private:
    class ContextConcept {
      public:
	virtual ~ContextConcept(){}
	virtual void print_name() const = 0;
	virtual void set_value(int i) = 0;
    };

    template<typename T>
    class ContextModel : public ContextConcept {
      public:
	ContextModel(T modelVal) : m_modelVal(modelVal) {}
	void print_name() const override { m_modelVal.print_name(); }
	void set_value(int i) override { m_modelVal.set_value(i); }
	T get() { return m_modelVal; }
      private:
	T m_modelVal;
    };

    std::shared_ptr<ContextConcept> m_value;
};

void func2(Context *con){
  std::cout << "func2 : ";
  con->print_name();
}

void func1(Context *con){
  func2(con);
}

int main(int argc, char*argv[])
{
  Context my_dev{GPU()};
  auto dev2 = my_dev;
  func1(&dev2);
  auto typed = my_dev.get<GPU>();
  typed.do_GPU();
  
  return 0;
}
