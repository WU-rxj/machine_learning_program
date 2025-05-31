#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>
#include<stdbool.h>

//出错的样例- ( 3 + 2 * 4 ) / 2


//存档，到期末后在做
//To do:变量赋值的覆盖问题
typedef enum{
    T_var,
    T_int,
    T_float,
    T_ope,
    T_error
}tt;

typedef struct{
    tt type;
    char str[32];
}Token;

typedef struct{
    char name[32];
    union{
        int ival;
        double fval;
    }val;
    enum{
        A_int,
        A_float
    }type;
}Assignment;

typedef struct value{
    enum{
        V_int,
        V_float,
        V_error
    }type;
    union{
        int ival;
        double fval;
    }val;
}Value;

Token tokens[2024];
int count1=0;

Assignment assign[200];
int count2=0;

bool is_variable(char *word);
bool is_integer(const char *word);
bool is_float(const char *word);
tt gettype(char *str);
Value getvalue(char *name);
void tokenize(char *input);
Value evalassign(int l, int r);
Value eval(int l, int r);
Value meetvalue(Value v1, Value v2, char op);
int check_p(int l, int r);
void print(Value v);
bool checkminus(int l,int r);
int findoperator(int l, int r);
void handle(char *token);

void handle(char *token) {
    int len = strlen(token);
    int w = 0;
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (token[i] == '-') {
            count++;
        } else {
            if (count % 2 == 1) {
                token[w++] = '-';
            }
            count = 0;
            token[w++] = token[i];
        }
    }
    if (count% 2 == 1) {
        token[w++] = '-';
    }
    token[w] = '\0';
}

bool is_variable(char *word) {
    if (!isalpha(*word) && *word!= '_') return false;
    while (*word) {
        if (!isalnum(*word) && *word!= '_') return false;
        word++;
    }
    return true;
}

bool is_integer(const char *word) {
    if (!*word) return false;
    while (*word) {
        if (!isdigit(*word)) return false;
        word++;
    }
    return true;
}

bool is_float(const char *word) {
    int sum=0;
    bool digit = false;
    while (*word) {
        if (*word=='.')
            sum++;
        else if (!isdigit(*word))
            return false;
        else {
            digit = true;
        }
        word++;
    }
    return (sum<=1) && digit;
}

tt gettype(char *str) {
    if (isalpha(str[0]) || str[0] == '_') {
        if (is_variable(str)){return T_var;}
        else {return T_error;}
    } else if (isdigit(str[0]) || str[0] == '.') {
        if(is_integer(str)){return T_int;}
        else if(is_float(str)){return T_float;}
        else {
            return T_error;
        }
    } else if (strchr("+-*/=()",str[0])) {
        return T_ope;
    } else {
        return T_error;
    }
}

void tokenize(char *input) {
    count1=0;
    char *token=strtok(input, " ");
    while (token!= NULL) {
        handle(token);
        Token t;
        t.type=gettype(token);
        strcpy(t.str,token);
        tokens[count1++]=t;
        token=strtok(NULL, " ");
    }
}

Value getvalue(char *name) {
    for (int i=0;i<count2;i++) {
        if (strcmp(assign[i].name,name) == 0) {
            Value v;
            if (assign[i].type==A_int) {
                v.type=V_int;
                v.val.ival=assign[i].val.ival;
            } else {
                v.type=V_float;
                v.val.fval=assign[i].val.fval;
            }
            return v;
        }
    }
    Value error;
    error.type = V_error;
    return error;
}

Value eval(int l, int r) {
    if (l > r) {
        Value v;
        v.type = V_error;
        return v;
    }

    if (l == r) {
        if (tokens[l].type == T_int) {
            Value v;
            v.type = V_int;
            v.val.ival = atoi(tokens[l].str);
            return v;
        } else if (tokens[l].type == T_float) {
            Value v;
            v.type = V_float;
            v.val.fval = atof(tokens[l].str);
            return v;
        } else if (tokens[l].type == T_var) {
            return getvalue(tokens[l].str);
        } else {
            Value v;
            v.type = V_error;
            return v;
        }
    }

    else if (check_p(l, r)) {
        return eval(l + 1, r - 1);
    }
    else if (checkminus(l, r) && findoperator(l + 1, r) == -1) {
        Value v = eval(l + 1, r);
        if (v.type == V_int) {
            v.val.ival = -v.val.ival;
        } else if (v.type == V_float) {
            v.val.fval = -v.val.fval;
        }
        return v;
    }

    else {
        int op = findoperator(l, r);
        if (op == -1) {
            Value v;
            v.type = V_error;
            return v;
        }
        Value v1 = eval(l, op - 1);
        Value v2 = eval(op + 1, r); 
        return meetvalue(v1, v2, tokens[op].str[0]);
    }
}


bool checkminus(int l, int r) {
    if (l <= r && tokens[l].str[0] == '-') {
        if (l == 0 || tokens[l - 1].type == T_ope || tokens[l - 1].str[0] == '(') {
            return true;
        }
    }
    return false;
}

int findoperator(int l, int r) {
    int op = -1;
    int kuo = 114514;
    int kuo1 = 0;

    for (int i = l; i <= r; i++) {
        if (tokens[i].type == T_ope) {
            if (tokens[i].str[0] == '(') {
                kuo1++;
            } else if (tokens[i].str[0] == ')') {
                kuo1--;
            } else if (kuo1 == 0) {
                int pre = (tokens[i].str[0] == '+' || tokens[i].str[0] == '-') ? 1 : 2;
                // 修复负号判断逻辑
                bool is_unary_minus = false;
                if (tokens[i].str[0] == '-') {
                    if (i == l || tokens[i - 1].str[0] == '+' || tokens[i - 1].str[0] == '-' ||
                        tokens[i - 1].str[0] == '*' || tokens[i - 1].str[0] == '/' ||
                        tokens[i - 1].str[0] == '(' || tokens[i - 1].str[0] == '=') {
                        is_unary_minus = true;
                    }
                }
                
                if (!is_unary_minus && pre <= kuo) {
                    op = i;
                    kuo = pre;
                }
            }
        }
    }
    return op;
}

Value meetvalue(Value v1, Value v2, char op) {
    if (v1.type == V_error || v2.type == V_error) {
        Value v;
        v.type = V_error;
        return v;
    }

    if (v1.type == V_int && v2.type == V_float) {
        v1.type = V_float;
        v1.val.fval = (double)v1.val.ival;
    } else if (v1.type == V_float && v2.type == V_int) {
        v2.type = V_float;
        v2.val.fval = (double)v2.val.ival;
    }

    Value v;
    switch (op) {
        case '+':
            if (v1.type == V_int && v2.type == V_int) {
                v.type = V_int;
                v.val.ival = v1.val.ival + v2.val.ival;
            } else {
                v.type = V_float;
                v.val.fval = v1.val.fval + v2.val.fval;
            }
            break;
        case '-':
            if (v1.type == V_int && v2.type == V_int) {
                v.type = V_int;
                v.val.ival = v1.val.ival - v2.val.ival;
            } else {
                v.type = V_float;
                v.val.fval = v1.val.fval - v2.val.fval;
            }
            break;
        case '*':
            if (v1.type == V_int && v2.type == V_int) {
                v.type = V_int;
                v.val.ival = v1.val.ival * v2.val.ival;
            } else {
                v.type = V_float;
                v.val.fval = v1.val.fval * v2.val.fval;
            }
            break;
        case '/':
            if (v1.type == V_int && v2.type == V_int) {
                v.type = V_int;
                v.val.ival = v1.val.ival / v2.val.ival;
            } else {
                v.type = V_float;
                v.val.fval = v1.val.fval / v2.val.fval;
            }
            break;
        default:
            v.type = V_error;
            break;
    }
    return v;
}


int check_p(int l,int r) {
    if (tokens[l].str[0]=='('&& tokens[r].str[0]==')') {
        int count=0;
        for (int i=l;i<=r;i++) {
            if (tokens[i].str[0]=='(') {count++;}
            if (tokens[i].str[0]==')') {count--;}
            if (count==0 && i<r) return 0;
        }
        if (count==0){return true;}
        else{return false;}
    }
    return 0;
}

Value evalassign(int l, int r) {
    // 处理特殊情况：以负号开头的赋值
    if (l < r && tokens[l].str[0] == '-' && l + 1 < r && tokens[l + 1].str[0] == '=') {
        Value error;
        error.type = V_error;
        return error;
    }
    
    // 从右到左查找等号，处理连续赋值
    for (int i = r - 1; i >= l + 1; i--) {
        if (tokens[i].str[0] == '=') {
            Value v = evalassign(i + 1, r);
            if (v.type == V_error) {
                return v;
            }
            
            if (tokens[i - 1].type == T_var) {
                int j;
                for (j = 0; j < count2; j++) {
                    if (strcmp(assign[j].name, tokens[i - 1].str) == 0) {
                        break;
                    }
                }
                if (j == count2) {
                    strcpy(assign[count2].name, tokens[i - 1].str);
                    count2++;
                }
                if (v.type == V_int) {
                    assign[j].type = A_int;
                    assign[j].val.ival = v.val.ival;
                } else {
                    assign[j].type = A_float;
                    assign[j].val.fval = v.val.fval;
                }
            }
            return v;
        }
    }
    return eval(l, r);
}

//时间超限再传结构体指针（ bug点 ）
void print(Value v) {
    if (v.type == V_int){
        printf("%d\n",v.val.ival);
    }else if(v.type==V_float){
        printf("%.6f\n",v.val.fval);
    }else{
        printf("Error\n");
    }
}

int main() {
    char input[2024];
    while (fgets(input,sizeof(input),stdin)!=NULL) {
        for (int i=0;input[i]!='\0';i++) {
            if (input[i]=='\n') {
                input[i]='\0';
                break;
            }
        }
        tokenize(input);
        Value result=evalassign(0,count1-1);
        print(result);
    }
    return 0;
}
