import { useState } from "react";
import { View, TextInput, Text, Pressable, ActivityIndicator, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import Icon from "react-native-vector-icons/Ionicons";
import { LinearGradient } from "expo-linear-gradient";

export default function Login() {
  const router = useRouter();
  const [form, setForm] = useState({ username: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const fetchWithTimeout = (url, options, timeout = 5000) => {
    return Promise.race([
      fetch(url, options),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Request timed out")), timeout)
      ),
    ]);
  };
  
  const handleLogin = async () => {
    if (!form.username || !form.password) {
      setErrorMessage("Username and password are required");
      return;
    }
  
    setLoading(true);
    setErrorMessage(null);
  
    try {
      const response = await fetchWithTimeout("http://10.10.119.148:8081/user/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
  
      const data = await response.json();
      console.log("Response:", response.status, data);
  
      setLoading(false);
      if (!response.ok) {
        setErrorMessage(data?.error || "Invalid credentials");
        return;
      }
  
      router.replace("/home");
    } catch (error) {
      setLoading(false);
      console.error("Fetch error:", error.message);
      setErrorMessage(error.message === "Request timed out" ? "Request timed out. Please try again." : "Network error. Please try again.");
    }
  };
  

  return (
<LinearGradient
    colors={["#1E3A8A", "#000000", ]}
      style={styles.container}
    >      <Pressable style={styles.backButton} onPress={() => router.push("/")}>
        <Icon name="arrow-back" size={24} color="white" />
        <Text style={styles.backText}>Back</Text>
      </Pressable>

      <Text style={styles.title}>Login</Text>

      <TextInput
        style={styles.input}
        placeholder="Username"
        placeholderTextColor="#888"
        autoCapitalize="none"
        value={form.username}
        onChangeText={(text) => setForm({ ...form, username: text })}
      />

      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor="#888"
        secureTextEntry
        value={form.password}
        onChangeText={(text) => setForm({ ...form, password: text })}
      />

     
{errorMessage && (
  <View style={styles.errorContainer}>
    <Icon name="warning-outline" size={18} color="#ff4d4d" style={styles.errorIcon} />
    <Text style={styles.errorMessage}>{errorMessage}</Text>
  </View>
)}


      <Pressable onPress={handleLogin} disabled={loading} style={[styles.button, loading && styles.buttonDisabled]}>
        {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Login</Text>}
      </Pressable>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
  }
,  
  backButton: {
    position: "absolute",
    top: 50,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
  },
  backText: {
    marginLeft: 5,
    fontSize: 16,
    color: "white",
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 32,
    textAlign: "center",
  },
  errorContainer: {
    flexDirection: "row", // Ensures icon and text are on the same line
    alignItems: "center", // Vertically aligns icon and text
    justifyContent: "center", // Center the content
    backgroundColor: "rgba(255, 77, 77, 0.2)", // Light red background
    padding: 10,
    borderRadius: 8,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#ff4d4d",
  },
  errorIcon: {
    marginRight: 8, // Space between icon and text
  },
  errorMessage: {
    color: "#ff4d4d",
    fontSize: 4,
    fontWeight: "600",
  },
  
  input: {
    borderWidth: 1,
    borderColor: "#ffffff",
    padding: 16,
    borderRadius: 12,
    width: "100%",
    marginBottom: 16,
    color: "#fff",
    backgroundColor: "rgba(255, 255, 255, 0.1)",
  },
  errorMessage: {
    color: "#ff4d4d",
    marginBottom: 16,
    fontSize: 14,
    textAlign: "center",
  },
  button: {
    backgroundColor: "#3b82f6",
    paddingVertical: 14,
    borderRadius: 12,
    width: "100%",
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
  },
});
